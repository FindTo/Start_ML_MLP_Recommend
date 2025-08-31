from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import desc
from table_post import Post
from table_user import User
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet
from typing import List
from datetime import datetime
from get_features_table import load_features, get_post_df
from get_predict_by_model import load_models
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# List of model features
columns = ['topic', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4',
            'cluster_5','cluster_6', 'cluster_7', 'cluster_8', 'cluster_9',
           'text_length', 'gender', 'age', 'country', 'exp_group', 'city_capital',
           'post_likes', 'post_views', 'hour', 'month', 'day', 'time_indicator',
           'main_topic_liked', 'main_topic_viewed', 'views_per_user',
           'likes_per_user']

user_df = load_features(os.getenv('USER_FEATURES_DF_NAME'))

post_df = load_features(os.getenv('POST_FEATURES_DF_NAME'))

nn_input_columns_df = load_features(os.getenv('NN_INPUT_COLUMNS_DF_NAME'))

post_original_df = get_post_df()

model = load_models()

app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db

@app.get("/user/{id}", response_model = UserGet)
def get_user(id: int, db: Session = Depends(get_db)):

    data = db.query(User).filter(User.id == id).first()

    if data == None:

        raise HTTPException(404, "user not found")

    else:
        logger.info(data)
        return data

@app.get("/post/{id}", response_model = PostGet)
def get_post(id: int, db: Session = Depends(get_db)):

    data = db.query(Post).filter(Post.id == id).first()

    if data == None:

        raise HTTPException(404, "post not found")

    else:

        return data

@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    data = db.query(Feed).filter(Feed.user_id == id).order_by(desc(Feed.time)).limit(limit).all()
    logger.info(data)

    return data

@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    return db.query(Feed).filter(Feed.post_id == id).order_by(desc(Feed.time)).limit(limit).all()

@app.get("/")
def ping():

    return "The MLP recommendation service is active"

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:

    # Finding user in the prepared df
    user_features = user_df[user_df['user_id'] == id].reset_index(drop=True)

    # Getting time-related features
    user_features['hour'] = time.hour
    user_features['day'] = time.day

    # Only 3 months (10, 11, 12) available

    if time.month == 11:

        user_features['month_11'] = 1.0
        user_features['month_12'] = 0.0

    elif time.month == 12:

        user_features['month_11'] = 0.0
        user_features['month_12'] = 1.0

    else:

        user_features['month_11'] = 0.0
        user_features['month_12'] = 0.0

    # Time indicator from the beginning of 2021 up to now
    user_features['time_indicator'] = (time.year - 2021) * 360 * 24 + time.month * 30 * 24 + time.day * 24 + time.hour

    #logger.info(user_features)

    # Post pull filtered by likes and views
    post_pull = post_df[(post_df['post_views'] > 80) & (post_df['post_likes'] > 12)]

    # Merge post pull with user vector and fill the gaps
    X = post_pull.combine_first(user_features)

    #logger.info(X.head())

    # Fill the gaps with user features from the first row
    for col in user_features.columns.to_list():
        X[col] = X[col].iloc[0]

    # Drop unnecessary IDs, indexes are by post df
    X.drop(['post_id', 'user_id'], axis=1, inplace=True)

    # Arrange column in accordance with NN input
    X = X[nn_input_columns_df['0'].to_list()]

    # Convert df to tensor and make predictions using NN model
    X_tens = torch.FloatTensor(X.values)
    X['ax'] = torch.sigmoid(model(X_tens)).detach().numpy().astype("float32")

    # Return post_id columns
    X = X.combine_first(post_pull)

    # First n=limit posts from pull with max like probability
    posts_recnd = X.sort_values(ascending=False, by='ax').head(limit)['post_id'].to_list()

    #logger.info(posts_recnd)

    posts_recnd_list = []

    # Making response by Pydantic using the obtained post IDs
    for i in posts_recnd:

        posts_recnd_list.append(PostGet(id=i,
                                        text=post_original_df[post_df['post_id'] == i].text.iloc[0],
                                        topic=post_original_df[post_df['post_id'] == i].topic.iloc[0])
                                )

    return posts_recnd_list
