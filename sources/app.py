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

user_df = load_features(os.getenv('USER_FEATURES_DF_NAME'))
user_df = user_df.set_index(user_df["user_id"].copy())

post_df = load_features(os.getenv('POST_FEATURES_DF_NAME'))
post_df = post_df.set_index(post_df["post_id"].copy())

nn_input_columns_df = load_features(os.getenv('NN_INPUT_COLUMNS_DF_NAME'))

post_original_df = get_post_df()
post_original_df = post_original_df.set_index(post_original_df["post_id"].copy())

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

    # Finding user in the prepared df - raise error if incorrect user value
    try:
        user_features = user_df.loc[id].copy()
    except KeyError:
        raise HTTPException(404, "User not found")

    # Getting time-related features
    user_features['hour'] = time.hour
    user_features['day'] = time.day
    # Only 3 months (10, 11, 12) available
    user_features["month_11"] = float(time.month == 11)
    user_features["month_12"] = float(time.month == 12)
    # Time indicator from the beginning of 2021 up to now
    user_features['time_indicator'] = (time.year - 2021) * 360 * 24 + time.month * 30 * 24 + time.day * 24 + time.hour

    # Post pool filtered by likes and views
    # post_pool = post_df[(post_df['post_views'] > 80) & (post_df['post_likes'] > 12)].copy()
    post_pool = post_df.copy()

    # Merge post pool with user vector and fill the gaps
    for col, val in user_features.items():
        post_pool[col] = val

    # Drop unnecessary IDs, indexes are by post df
    X = post_pool.drop(['post_id', 'user_id'], axis=1)

    # Arrange columns in accordance with NN input
    X = X[nn_input_columns_df['0'].to_list()]

    # Convert df to tensor and make predictions using NN model
    X_tens = torch.tensor(X.values, dtype=torch.float32)

    # Check device and send tensor to it
    device = next(model.parameters()).device
    X_tens = X_tens.to(device)

    # Make predictions
    with torch.no_grad():
        scores  = torch.sigmoid(model(X_tens)).cpu().numpy().astype("float32")

    post_pool['ax'] = scores

    # First n=limit posts from pool with the max like probability
    top_posts = post_pool.nlargest(limit, 'ax')[['post_id']]

    # Making response by Pydantic using the obtained post IDs
    results = [
        PostGet(id=int(pid), text=row["text"], topic=row["topic"])
        for pid, row in post_original_df.loc[top_posts["post_id"]].iterrows()
    ]

    return results
