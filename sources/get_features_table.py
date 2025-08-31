import pandas as pd
from learn_model import get_user_df
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_post_df():

    # Obtain db connection
    post = pd.read_sql("SELECT * FROM public.post_text_df;", os.getenv('DATABASE_URL'))
    print(post.head())
    return post

# Obtain low-weight tables for all posts and users with features for NN input + listed NN input names
def get_user_post_features() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    # Read big DF from NN learning, with all features concatenated
    data = pd.read_csv('df_to_learn_mlp_128d_post_embedd.csv', sep=';')
    # data = load_features()

    print(data.shape)
    print(data.head())
    print(data.columns)

    # Saving original list of input features for NN apply
    list_of_columns_learning = data.columns.to_list()
    df_columns_learning = pd.DataFrame(list_of_columns_learning).drop([0, 1, 2], axis=0)

    # Choosing features to be dropped in DF from learning
    list_to_drop_user = ['post_id',
                         'text_length',
                         'target',
                         'gender',
                         'age',
                         'city_capital',
                         'post_likes',
                         'post_views',
                         'hour',
                         'day',
                         'time_indicator',
                         'country_Belarus',
                         'country_Cyprus',
                         'country_Estonia',
                         'country_Finland',
                         'country_Kazakhstan',
                         'country_Latvia',
                         'country_Russia',
                         'country_Switzerland',
                         'country_Turkey',
                         'country_Ukraine',
                         'exp_group_1',
                         'exp_group_2',
                         'exp_group_3',
                         'exp_group_4',
                         'topic_covid',
                         'topic_entertainment',
                         'topic_movie',
                         'topic_politics',
                         'topic_sport',
                         'topic_tech',
                         'month_11',
                         'month_12'
                         ]

    for i in range(128):
        list_to_drop_user.append(f'embed_{i}')

    # Drop post-related and time-related features
    user_df_new = data.drop(list_to_drop_user, axis=1)

    # Download original user table
    user = get_user_df()
    print(user.user_id.nunique())

    # Updating 'city' feature to 'city_capital'
    capitals = ['Moscow', 'Saint Petersburg', 'Kyiv', 'Minsk', 'Baku', 'Almaty', 'Astana', 'Helsinki',
                'Istanbul', 'Ankara', 'Riga', 'Nicosia', 'Limassol', 'Zurich', 'Bern', 'Tallin']
    user['city'] = user.city.apply(lambda x: 1 if x in capitals else 0)
    user = user.rename(columns={"city": "city_capital"})

    # Drop unnecessary features
    user = user.drop(['os', 'source'], axis=1)

    # One-hot encoding for categorial features in user table
    user_categorial_columns = ['country', 'exp_group']

    for col in user_categorial_columns:

        one_hot = pd.get_dummies(user[col], prefix=col, drop_first=True, dtype='int32')
        user = pd.concat((user.drop(col, axis=1), one_hot), axis=1)

    # Convert user to float32
    user = user.astype('float32')

    # Creating full DF with all users and features
    df_user_full = user.merge(user_df_new, on='user_id', how='left')
    df_user_full = df_user_full.drop_duplicates()
    print(df_user_full.head())

    # Fill empty features after merge with modes/mean values
    new_user_columns = ['main_topic_viewed_covid',
                        'main_topic_viewed_entertainment', 'main_topic_viewed_movie',
                        'main_topic_viewed_politics', 'main_topic_viewed_sport',
                        'main_topic_viewed_tech', 'main_topic_liked_covid',
                        'main_topic_liked_entertainment', 'main_topic_liked_movie',
                        'main_topic_liked_politics', 'main_topic_liked_sport',
                        'main_topic_liked_tech']

    for col in new_user_columns:
        df_user_full[col] = df_user_full[col].fillna(df_user_full[col].mode()[0])

    df_user_full['views_per_user'] = df_user_full['views_per_user'].fillna(df_user_full['views_per_user'].mean())
    df_user_full['likes_per_user'] = df_user_full['likes_per_user'].fillna(df_user_full['likes_per_user'].mean())

    total_memory = df_user_full.memory_usage(deep=True).sum()
    print(f"\ndf_user_full total memory: {total_memory} bytes")

    # Read 'post' df with embedding-based PCA features
    post = pd.read_csv('post_with_200xPCA_embed_1024k.csv', sep=';')

    # One-hot encoding for 'topic' feature
    one_hot = pd.get_dummies(post['topic'], prefix='topic', drop_first=True, dtype='float32')
    post = pd.concat((post.drop('topic', axis=1), one_hot), axis=1)

    print(post.head())

    # Choosing post-related features from big learning df
    df_post_new = data[['post_id', 'post_likes', 'post_views']].drop_duplicates()

    # Merge feed-based post features with original DF with PCA
    df_post_full = post.merge(df_post_new, on='post_id', how='left')

    # Fill empty values with mean
    df_post_full['post_likes'] = df_post_full['post_likes'].fillna(df_post_full['post_likes'].mean())
    df_post_full['post_views'] = df_post_full['post_views'].fillna(df_post_full['post_views'].mean())

    total_memory = df_post_full.memory_usage(deep=True).sum()
    print(f"\nPost DF total memory: {total_memory} bytes")

    return df_user_full, df_post_full, df_columns_learning

# Send DF to the DB  - no chunks
def df_to_sql(df, name):

    # Try to write DF into the db by chunks
    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    try:

        print((f"to_sql - start writing {name}"))
        df.to_sql(name,
                  con=engine,
                  if_exists='replace',
                  index=False)
        print((f"to_sql - {name} successfully written"))
        conn.close()

    except Exception as e:

        print(f"to_sql - failed to write {name}")
        raise RuntimeError(f"Loading error: {e}")

    return 0


def csv_to_sql(csv_name,table_name):


    # Try to write csv file into the db by chunks
    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    try:

        print((f"to_sql - start writing {table_name}"))

        chunksize = int(os.getenv('CHUNKSIZE'))

        for chunk in pd.read_csv(csv_name, chunksize=chunksize):

            chunk.to_sql(table_name, engine, if_exists='append', index=False, method='multi')

        print((f"to_sql - {table_name} successfully written"))
        conn.close()

    except Exception as e:

        print(f"to_sql - failed to write {table_name}")
        raise RuntimeError(f"Loading error: {e}")

    finally:
        conn.close()

    return 0


# Load DF with features from DB using chunks
def load_features(features_name) -> pd.DataFrame:

    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    try:

        print((f"from sql - start loading {features_name}"))
        for chunk_dataframe in pd.read_sql(features_name,
                                           conn, chunksize=int(os.getenv('CHUNKSIZE'))):

            chunks.append(chunk_dataframe)

        print((f"from sql - {features_name} loaded successfully"))

    except Exception as e:

        raise RuntimeError(f"Loading error: {e}")

    finally:
        conn.close()

    return pd.concat(chunks, ignore_index=True)

