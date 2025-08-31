import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, auc, accuracy_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

# Download transformed BERT embeddings for posts (128D) from SQL database or from local
def get_embedd_df(is_csv=False, sep=';'):

    if is_csv:
        embedds = pd.read_csv('df_post_128d_embedd_with_id_pure.csv', sep=sep)
    else:
        # Loading post embedd
        embedds = pd.read_sql(f"SELECT * FROM {os.getenv('EMBEDD_DF_NAME')};", os.getenv('DATABASE_URL'))

    print(embedds.head())
    return embedds

# Autoencoder class - for reducing space of  BERT-like embeddings
class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, latent_dim)  # 128D latent space
        )
        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# Class for post's data: separate data an ID
class Post_Data(Dataset):
    def __init__(self, data_source, is_csv=False, sep=';'):
        if is_csv:
            df = pd.read_csv(data_source, sep=sep)
        else:
            if not isinstance(data_source, pd.DataFrame):
                raise TypeError("If is_csv=False => data_source must be pd.DataFrame")
            df = data_source.copy()

        self.post_id = df['post_id']
        self.data = df.drop(['post_id'], axis=1)

    def __getitem__(self, idx):
        vector = self.data.loc[idx]
        post_id = self.post_id.loc[idx]

        vector = torch.FloatTensor(vector)
        post_id = torch.FloatTensor([post_id])

        return vector, post_id

    def __len__(self):
        return len(self.post_id)


def autoencoder_plot_stats(
        history: dict,
        title: str,
):
    plt.figure(figsize=(12, 4))

    # График Loss и MAE
    plt.subplot(1, 2, 1)
    plt.title(title + ' loss')
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.xlabel("Epoch")
    plt.legend()

    # График процентилей MAE
    plt.subplot(1, 2, 2)
    plt.title(title + ' Percentiles: 25-50-75')
    plt.plot(history["train_mae_25p"], label="25th train")
    plt.plot(history["train_mae_median"], label="50th train")
    plt.plot(history["train_mae_75p"], label="75th train")
    plt.plot(history["test_mae_25p"], label="25th test")
    plt.plot(history["test_mae_median"], label="50th test")
    plt.plot(history["test_mae_75p"], label="75th test")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

# MAE accuracy for autoencoder training visualisation
def mae_accuracy(preds, x, is_test=False):
    mae = torch.abs(preds - x).mean(dim=1)  # MAE per sample
    mae_np = mae.cpu().detach().numpy()

    if not is_test:
        # return varios MAE metrics per batch
        return {
            "train_mae_mean": np.mean(mae_np),
            "train_mae_median": np.median(mae_np),
            "train_mae_25p": np.percentile(mae_np, 25),
            "train_mae_75p": np.percentile(mae_np, 75),
        }
    else:
        return {
            "test_mae_mean": np.mean(mae_np),
            "test_mae_median": np.median(mae_np),
            "test_mae_25p": np.percentile(mae_np, 25),
            "test_mae_75p": np.percentile(mae_np, 75),
        }

# Autoencoder inference mode - to get metrics
@torch.inference_mode()
def autoencoder_evaluate(model, loader, loss_fn, history):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    epoch_loss = 0
    epoch_mae_metrics = {"test_mae_mean": [], "test_mae_median": [], "test_mae_25p": [], "test_mae_75p": []}

    for x, _ in tqdm(loader, desc='Test'):
        x = x.to(device)

        output = model(x)[0]

        loss = loss_fn(output, x)

        epoch_loss += loss.item()

        accuracy = mae_accuracy(output.detach().cpu(), x.detach().cpu(), is_test=True)

        for k in epoch_mae_metrics:
            epoch_mae_metrics[k].append(accuracy[k])

    history["test_loss"].append(epoch_loss / len(loader))

    for k in epoch_mae_metrics:
        history[k].append(np.mean(epoch_mae_metrics[k]))

    return history


# Train autoencoder using post's BERT-like embeddings - the whole cycle
def autoencoder_train(data_source, is_csv=False, sep=';', lr=1e-3, n_epoch=20):
    history = {
        "train_loss": [],
        "train_mae_mean": [],
        "train_mae_median": [],
        "train_mae_25p": [],
        "train_mae_75p": [],
        "test_loss": [],
        "test_mae_mean": [],
        "test_mae_median": [],
        "test_mae_25p": [],
        "test_mae_75p": [],

    }

    # Create dataset
    dataset = Post_Data(data_source,is_csv=is_csv, sep=sep)
    train_dataset, test_dataset = random_split(dataset,
                                               (int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8))
                                               )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

    # Create the model
    model = Autoencoder(input_dim=768, latent_dim=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # MAE loss
    model.train()

    for epoch in range(n_epoch):

        epoch_loss = 0
        epoch_mae_metrics = {"train_mae_mean": [],
                             "train_mae_median": [],
                             "train_mae_25p": [],
                             "train_mae_75p": []
                             }

        for x, _ in tqdm(train_loader, desc='Train'):
            x = x.to(device)

            optimizer.zero_grad()

            output = model(x)[0]

            loss = loss_fn(output, x)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            accuracy = mae_accuracy(output.detach().cpu(), x.detach().cpu())

            for k in epoch_mae_metrics:
                epoch_mae_metrics[k].append(accuracy[k])

        history["train_loss"].append(epoch_loss / len(train_loader))

        for k in epoch_mae_metrics:
            history[k].append(np.mean(epoch_mae_metrics[k]))

        autoencoder_evaluate(model, test_loader, loss_fn, history)

        clear_output()
        autoencoder_plot_stats(history, 'Posts autoencoder 768->128')

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Test Loss: {history['test_loss'][-1]:.4f} | "
            f"Train MAE: {history['train_mae_mean'][-1]:.4f} (median: {history['train_mae_median'][-1]:.4f}) | "
            f"Test MAE: {history['test_mae_mean'][-1]:.4f} (median: {history['test_mae_median'][-1]:.4f}) | "
            f"Train 25-75p: [{history['train_mae_25p'][-1]:.4f}, {history['train_mae_75p'][-1]:.4f}]"
            f"Test 25-75p: [{history['test_mae_25p'][-1]:.4f}, {history['test_mae_75p'][-1]:.4f}]"
        )

    torch.save(model.state_dict(), 'post_autoencoder_drop_0_3_0_2_pure.pt')

    return model, history

# Dataset for posts recommendation learning
class Recommend_Data(Dataset):
    def __init__(self, df):

        self.target = df['target'].to_numpy(dtype='float32')
        self.data = df.drop(['target', 'user_id', 'post_id'], axis=1).to_numpy(dtype='float32')
        self.len = len(self.target)

    def __getitem__(self, idx):
        vector = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor([self.target[idx]], dtype=torch.float32)
        return vector, target

    def __len__(self):
        return self.len

# FC NN for classification
def create_nn_to_classify():
    class MLP_Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Sequential(

                nn.Linear(173, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )

            self.fc2 = nn.Sequential(

                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )

            self.fc3 = nn.Sequential(

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )

            self.output = nn.Sequential(

                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(16, 1),
            )


        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x) + x
            x = self.fc3(x)
            return self.output(x)

    return MLP_Model()


# Choose the best probability threshold for max accuracy
def find_best_threshold(y_true, y_pred_probs):
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_acc = 0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    print("Best threshold for accuracy:", best_thresh)
    return best_thresh

# Binary accuracy and ROC calculation
def _metrics(preds, y):
    y_true = np.concatenate(y)
    y_pred = torch.sigmoid(torch.from_numpy(np.concatenate(preds))).numpy()
    y_pred_label = (y_pred >= find_best_threshold(y_true, y_pred )).astype(float)
    acc = accuracy_score(y_true, y_pred_label)
    roc = roc_auc_score(y_true, y_pred)
    return acc, roc


# Train cycle of NN
def train(model, train_loader, device, optimizer, loss_fn)-> tuple[float, float, float]:
    model.train()
    train_loss = 0
    y_true_list = []
    logits_list = []

    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        train_loss += loss.item()

        y_true_list.append(y.detach().cpu().numpy())
        logits_list.append(output.detach().cpu().numpy())

        loss.backward()

        optimizer.step()

    train_loss /= len(train_loader)
    train_accuracy, train_roc = _metrics(logits_list, y_true_list)

    return train_loss, train_accuracy, train_roc

# Calculate NN output and estimate accuracy
@torch.inference_mode()
def evaluate(model, loader, device, loss_fn) -> tuple[float, float, float]:
    model.eval()
    test_loss = 0
    y_true_list = []
    logits_list = []

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.to(device), y.to(device)

        output = model(x)

        y_true_list.append(y.detach().cpu().numpy())
        logits_list.append(output.detach().cpu().numpy())

        loss = loss_fn(output, y)

        test_loss += loss.item()

    test_loss /= len(loader)
    test_accuracy, test_roc = _metrics(logits_list, y_true_list)

    return test_loss, test_accuracy, test_roc

# Plot graphs of accuraccy and loss change dynamic during epochs
def plot_stats(
        train_loss: list[float],
        valid_loss: list[float],
        train_accuracy: list[float],
        valid_accuracy: list[float],
        train_roc: list[float],
        valid_roc: list[float],
        title: str
):
    plt.figure(figsize=(16, 8))

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + ' accuracy')

    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(valid_accuracy, label='Valid accuracy')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + ' ROC-AUC')

    plt.plot(train_roc, label='Train ROC-AUC')
    plt.plot(valid_roc, label='Valid ROC-AUC')
    plt.legend()
    plt.grid()

    plt.show()


# Learn NN for the specified number of epochs
def whole_train_valid_cycle(model,
                            num_epochs,
                            title,
                            train_loader,
                            test_loader,
                            device,
                            optimizer):
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []
    train_roc_history, valid_roc_history = [], []

    # For imbalanced classes - 20% of likes - set pos_weight
    pos_weight = torch.tensor([4.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_roc = train(model, train_loader, device, optimizer, loss_fn)
        valid_loss, valid_accuracy, valid_roc = evaluate(model, test_loader, device, loss_fn)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

        train_roc_history.append(train_roc)
        valid_roc_history.append(valid_roc)

        clear_output()

        plot_stats(
            train_loss_history, valid_loss_history,
            train_accuracy_history, valid_accuracy_history,
            train_roc_history, valid_roc_history,
            title
        )

# Download user's data from the SQL database
def get_user_df():

    # Установка соединения с базой данных
    user = pd.read_sql("SELECT * FROM public.user_data;", os.getenv('DATABASE_URL'))
    print(user.head())
    return user

# Download posts data from the SQL database
def get_post_df():

    # Установка соединения с базой данных
    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post

# Download transformed BERT embeddings for posts (128D) from SQL database or from local
def get_embedd_df(is_csv=False, sep=';'):

    if is_csv:
        embedds = pd.read_csv('df_post_128d_embedd_with_id_pure.csv', sep=sep)
    else:
        # Loading post embedd
        embedds = pd.read_sql(f"SELECT * FROM {os.getenv('EMBEDD_DF_NAME')};", os.getenv('DATABASE_URL'))

    print(embedds.head())
    return embedds

# Obtaining DF with vector of all features for NN learning
def get_vector_df(df_embedd:pd.DataFrame, feed_n_lines=1024000):

    # Установка соединения с базой данных
    user = get_user_df()
    post = get_post_df()
    feed = pd.read_sql(f"SELECT * FROM public.feed_data order by random() LIMIT {feed_n_lines};", os.getenv('DATABASE_URL'))
    feed = feed.drop_duplicates()
    print(feed.head())

    # Поработаем с категориальными колонками для таблицы new_user. Колонку exp_group тоже считаем как категориальную

    new_user = user.drop('city', axis=1)

    categorical_columns = []
    categorical_columns.append('country')
    # categorical_columns.append('os')
    # categorical_columns.append('source')
    categorical_columns.append('exp_group')  # разобью по группам категориальный признак

    # Добавил булевый признак по главным городам в представленных странах, остальных городов слишком много
    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    cap_bool = user.city.apply(lambda x: 1 if x in capitals else 0)

    # добавил признак по главным городам в представленных странах
    new_user = pd.concat([new_user, cap_bool], axis=1, join='inner')
    new_user = new_user.rename(columns={"city": "city_capital"})

    # Выбираем только числовые столбцы для преобразования
    numeric_columns = new_user.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns

    # Преобразуем только числовые столбцы в float32
    new_user[numeric_columns] = new_user[numeric_columns].astype('float32')

    # готовая таблица User для обучения
    new_user.head()
    num_user_full = new_user['user_id'].nunique()
    print(f'Число уникальных юзеров:{num_user_full}')

    # Длина текста поста - новый признак для таблицы Post
    post['text_length'] = post['text'].apply(len)
    # post = post.rename(columns={"text": "text_feature"})

    # Убираем исходные тексты из признаков
    post = post.drop(['text'], axis=1)

    # Конкатенирую с топом эмбеддингов, максимально вносящих вклад в PCA фичи
    post = post.merge(df_embedd,on='post_id', how='left')

    print(post.head())

    # разобью по группам категориальный признак из Post
    categorical_columns.append('topic')

    # Выбираем только числовые столбцы таблицы Post для преобразования
    numeric_columns = post.select_dtypes(include=['float64', 'int64']).columns

    # Преобразуем только числовые столбцы в float32
    post[numeric_columns] = post[numeric_columns].astype('float32')

    # Выбираем только числовые столбцы таблицы Feed для преобразования
    numeric_columns = feed.select_dtypes(include=['float64', 'int64']).columns

    # Преобразуем только числовые столбцы в float32
    feed[numeric_columns] = feed[numeric_columns].astype('float32')

    # Ренейм action на случай пересечений с колонками TD-IDF
    feed = feed.rename(columns={"action": "action_class"})

    # Теперь нужно объединить все с таблицей Feed, чтобы получить мастер-таблицу для трейна

    df = pd.merge(
        feed,
        post,
        on='post_id',
        how='left'
    )

    df = pd.merge(
        df,
        new_user,
        on='user_id',
        how='left'
    )
    df.head()

    # Признак-счетчик лайков для постов
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)
    df['post_likes'] = df.groupby('post_id')['action_class'].transform('sum')

    # Признак-счетчик просмотров для постов
    # df['views_per_post'] = df.groupby('post_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['post_views'] = df.groupby('post_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Поправим Datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Нужно отсортировать по time для валидации, по возрастанию
    df = df.sort_values('timestamp')

    # Набираю признаки из timestamp
    df['day_of_week'] = df.timestamp.dt.dayofweek
    df['hour'] = df.timestamp.dt.hour
    df['month'] = df.timestamp.dt.month
    df['day'] = df.timestamp.dt.day
    df['year'] = df.timestamp.dt.year

    # Фича-индикатор суммарного времени с 2021 года до текущего момента просмотра, в часах
    df['time_indicator'] = (df['year'] - 2021) * 360 * 24 + df['month'] * 30 * 24 + df['day'] * 24 + df['hour']

    categorical_columns.append('month')  # разобью по группам month  из Feed

    # Генерим фичи: топ topic для пользователей из feed по лайкам/просмотрам
    main_liked_topics = df[df['action_class'] == 1].groupby(['user_id'])['topic'].agg(
        lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_liked_topics = main_liked_topics.rename(columns={"topic": "main_topic_liked"})
    main_viewed_topics = df[df['action_class'] == 0].groupby(['user_id'])['topic'].agg(
        lambda x: np.random.choice(x.mode())).to_frame().reset_index()
    main_viewed_topics = main_viewed_topics.rename(columns={"topic": "main_topic_viewed"})

    # Присоединяем к мастер-таблице
    df = pd.merge(df, main_liked_topics, on='user_id', how='left')
    df = pd.merge(df, main_viewed_topics, on='user_id', how='left')

    # Заполняем пропуски самой частой категорией
    df['main_topic_liked'].fillna(df['main_topic_liked'].mode().item(), inplace=True)
    df['main_topic_viewed'].fillna(df['main_topic_viewed'].mode().item(), inplace=True)

    # Разобью по группам категориальный признак из Feed
    categorical_columns.append('main_topic_viewed')
    categorical_columns.append('main_topic_liked')

    # Признак-счетчик лайков по юзерам
    likes_per_user = df.groupby(['user_id'])['action_class'].agg(pd.Series.sum).to_frame().reset_index()
    likes_per_user = likes_per_user.rename(columns={"action_class": "likes_per_user"})

    # Признак-счетчик просмотров для юзеров
    # df['views_per_user'] = df.groupby('user_id')['action_class'].apply(lambda x: 1 if x == 0 else 0).transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 0 if x == 'like' or x == 1 else 1)
    df['views_per_user'] = df.groupby('user_id')['action_class'].transform('sum')
    df['action_class'] = df.action_class.apply(lambda x: 1 if x == 'like' or x == 1 else 0)

    # Присоединяем к мастер-таблице
    df = pd.merge(df, likes_per_user, on='user_id', how='left')

    # Выбираем только числовые столбцы для преобразования
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    df[numeric_columns] = df[numeric_columns].astype('float32')

    num_user_df = df['user_id'].nunique()
    print(f'Число уникальных юзеров в итоговом датасете:{num_user_df}')

    num_post_df = df['post_id'].nunique()
    print(f'Число уникальных постов в итоговом датасете:{num_post_df}')

    # В датасете есть повторения, где у таргета и action_class несогласованны данные.
    # При этом это одна и та же запись, по сути. Не буду убирать дублеры.
    # Просто задам таргету 1 если у строки был лайк. Тем самым данные не будут противоречить друг другу
    df['target'] = df['target'].astype('int32')
    df['action_class'] = df['action_class'].astype('int32')
    df['target'] = df['target'] | df['action_class']

    # Уберем лишние признаки
    df = df.drop(['timestamp', 'action_class', 'os', 'source', 'day_of_week', 'year'], axis=1)

    # Преобразуем численные категориальные в int32
    df[['exp_group', 'month']] = df[['exp_group', 'month']].astype('int32')

    print(categorical_columns)

    # One-hot encoding для всех категориальных колонок
    for col in categorical_columns:
        one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype='int32')

        df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)

    # Оставляю user_id и post_id, их нужно будет дропнуть при создании датасета
    # df = df.drop(['user_id', 'post_id'], axis=1)

    df = df.astype('float32')
    print('Итоговый датасет:')
    print(df.head)

    # сохраняю с user_id для генерации таблицы признаков для сервера
    df.to_csv('df_to_learn_mlp_128d_post_embedd.csv', sep=';', index=False)

    # Получение общего объема памяти, занимаемой DataFrame
    total_memory = df.memory_usage(deep=True).sum()
    print(f"\nОбщий объем памяти, занимаемой DataFrame: {total_memory} байт")
    print(df.dtypes)

    return df, categorical_columns, post


