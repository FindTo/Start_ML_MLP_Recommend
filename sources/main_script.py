from get_features_table import df_to_sql, get_user_post_features, load_features
from get_post_embeddings import make_roberta_embeddings,Post_Data,get_128d_embeddings
from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, auc, accuracy_score
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load env variables
load_dotenv()

IS_NEW_MODEL=False
N_EPOCHS=3
LR=25e-5
WEIGHT_DEC=1e-4
BATCH_SIZE=256
MODEL_NAME = 'nn_estinmate_likes_128d_embedds_1024k_drop_03_02.pt'

if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader, random_split
    from torch.optim import Adam
    from learn_model import (autoencoder_train,
                             get_embedd_df,
                             get_vector_df,
                             Recommend_Data,
                             create_nn_to_classify,
                             whole_train_valid_cycle)
    import pandas as pd
    #
    # # Load RoBerta and prepare text embeddings 768d
    # df_post_embed = make_roberta_embeddings()
    #
    # # Train autoencoder
    # autoencoder_model, _ = autoencoder_train(df_post_embed)
    #
    # # Create dataset object form post embeddings dataframe
    # post_dataset = Post_Data(df_post_embed)
    #
    # # Receive 128d embeddings
    # df_post_128d = get_128d_embeddings(autoencoder_model, post_dataset)

    # df_post_128d = get_embedd_df(is_csv=True)

    # Наберем необходимые записи
    # data, cat_columns, post = get_vector_df(df_post_128d, is_csv=True)
    data = pd.read_csv('df_to_learn_mlp_128d_post_embedd.csv', sep=';')

    # Dataset init from class
    dataset = Recommend_Data(data)

    generator = torch.Generator().manual_seed(123)

    # Datasest random split, 20% for test
    train_dataset, test_dataset = random_split(dataset,
                                               (int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)),
                                               generator=generator)

    # 128 batch size - optimal (empyrical)
    # Create loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              pin_memory=True,
                              shuffle=True
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             pin_memory=True,
                             shuffle=False)

    # Choosing device: Cuds if presented unless CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    # Creating NN for learning
    model = create_nn_to_classify()

    # If model is presented locally
    if not IS_NEW_MODEL:

        model.load_state_dict(torch.load(
            MODEL_NAME,
            weights_only=True))

    model = model.to(device)

    # Adam optimizer with default learning rate
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)

    whole_train_valid_cycle(model,
                            N_EPOCHS,
                            'Learning post recommendations',
                            train_loader,
                            test_loader,
                            device,
                            optimizer)

    torch.save(model.state_dict(), MODEL_NAME)

    # Estimate accuracy, F-measure and AUC based on the whole dataset

    X = data.drop(['target', 'user_id', 'post_id'], axis=1)
    y = data.target
    model.eval().cpu()

    X_tens = torch.FloatTensor(X.values)
    F_X = torch.round(torch.sigmoid(model(X_tens))).detach().numpy().astype("float32")
    Prob_X = torch.sigmoid(model(X_tens)).detach().numpy().astype("float32")

    # F-measure
    f1_loc_tr = round(f1_score(y,
                               F_X,
                               average='weighted'), 5)
    print(f'F-measure for FC NN: {f1_loc_tr}')

    # AUC
    fpr, tpr, thd = roc_curve(y, Prob_X)
    print(f'AUC for FC NN: {auc(fpr, tpr):.5f}')

    acc = accuracy_score(y, F_X)
    print(f'Accuracy for FC NN: {acc:.5f}')

#
    # user_df, post_df, nn_input_columns_df = get_user_post_features()
#
# df_to_sql(user_df, os.getenv('USER_FEATURES_DF_NAME'))
# df_to_sql(post_df, os.getenv('POST_FEATURES_DF_NAME'))
# df_to_sql(nn_input_columns_df, os.getenv('NN_INPUT_COLUMNS_DF_NAME'))


