# src/data_preprocess.py
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    reviews_path = RAW_DIR / "yelp_academic_dataset_review.csv"
    rvvec_path = RAW_DIR / "review_vectors.npy"
    business_path = RAW_DIR / "LDA_theta_synced.csv"

    reviews = pd.read_csv(reviews_path)
    review_vectors = np.load(rvvec_path)
    business_df = pd.read_csv(business_path)
    business_df = business_df.iloc[:, 0:-1]

    return reviews, review_vectors, business_df

'''
def build_index_maps(reviews, business_df):
    # 保证 user_id, business_id 为 string/int 一致处理
    reviews = reviews.copy()
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["business_id"] = reviews["business_id"].astype(str)
    business_df["business_id"] = business_df["business_id"].astype(str)

    # 添加 review_index（使用自然行索引）
    reviews = reviews.reset_index(drop=True)
    reviews["review_index"] = reviews.index.astype(int)

    # user / business 映射为连续索引
    unique_users = reviews["user_id"].unique().tolist()
    unique_businesses = reviews["business_id"].unique().tolist()

    user2idx = {u: i for i, u in enumerate(unique_users)}
    business2idx = {b: i for i, b in enumerate(unique_businesses)}

    # map columns
    reviews["user_idx"] = reviews["user_id"].map(user2idx).astype(int)
    reviews["business_idx"] = reviews["business_id"].map(business2idx).astype(int)

    # reindex business_df to only include businesses present in reviews (and keep original business features)
    business_df = business_df.set_index("business_id").reindex(unique_businesses).reset_index()

    return reviews, business_df, user2idx, business2idx
'''
def build_index_maps(reviews, business_df):
    # 保证 user_id, business_id 为 string/int 一致处理
    reviews = reviews.copy()
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["business_id"] = reviews["business_id"].astype(str)
    business_df["business_id"] = business_df["business_id"].astype(str)

    # 只保留在 business_df 里有特征的 business
    valid_business_ids = set(business_df["business_id"])
    reviews = reviews[reviews["business_id"].isin(valid_business_ids)].reset_index(drop=True)

    # 添加 review_index（使用自然行索引）
    reviews["review_index"] = reviews.index.astype(int)

    # user / business 映射为连续索引
    unique_users = reviews["user_id"].unique().tolist()
    unique_businesses = reviews["business_id"].unique().tolist()

    user2idx = {u: i for i, u in enumerate(unique_users)}
    business2idx = {b: i for i, b in enumerate(unique_businesses)}

    # map columns
    reviews["user_idx"] = reviews["user_id"].map(user2idx).astype(int)
    reviews["business_idx"] = reviews["business_id"].map(business2idx).astype(int)

    # 只保留在 reviews 里出现过的 business，并按 unique_businesses 顺序排列
    business_df = business_df.set_index("business_id").reindex(unique_businesses).reset_index()

    return reviews, business_df, user2idx, business2idx

def split_interactions(reviews, test_size=0.1, val_size=0.1, random_state=42):
    """
    将原始交互拆成 train / valid / test
    首先从全样本中抽出 test（test_size），然后从剩下的抽出 valid（val_size/(1-test_size)）
    这是按交互随机划分的简单方法。
    """
    df = reviews.copy()
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    # 对 train_val 再拆出 valid
    val_relative_size = val_size / (1.0 - test_size)
    train, valid = train_test_split(train_val, test_size=val_relative_size, random_state=random_state, shuffle=True)
    # reset index for neatness
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, valid, test


def save_processed(train, valid, test, review_vectors, business_df, user2idx, business2idx):
    # 保存 csv / npy / json
    train.to_csv(PROC_DIR / "train.csv", index=False)
    valid.to_csv(PROC_DIR / "valid.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)

    # business features 按 business_idx 排序保存为 npy（行 i 对应 business_idx = i）
    # 我们假设 business_df 的 columns = ['business_id', f1, f2, ...]
    feat_cols = [c for c in business_df.columns if c != "business_id"]
    business_feats = business_df[feat_cols].to_numpy(dtype=np.float32)
    np.save(PROC_DIR / "business_features.npy", business_feats)

    # 保存 review_vectors 以备训练时直接载入（不改变）
    np.save(PROC_DIR / "review_vectors.npy", review_vectors)

    # 保存映射
    with open(PROC_DIR / "user2idx.json", "w", encoding="utf8") as f:
        json.dump(user2idx, f, ensure_ascii=False)
    with open(PROC_DIR / "business2idx.json", "w", encoding="utf8") as f:
        json.dump(business2idx, f, ensure_ascii=False)

    meta = {
        "n_users": len(user2idx),
        "n_businesses": len(business2idx),
        "n_reviews": int(train.shape[0] + valid.shape[0] + test.shape[0])
    }
    with open(PROC_DIR / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)


def run_preprocess(test_size=0.1, val_size=0.1, random_state=42):
    print("Loading raw data...")
    reviews, review_vectors, business_df = load_raw()
    print(f"Raw reviews: {len(reviews)}, review_vectors shape: {review_vectors.shape}, business: {len(business_df)}")

    print("Building index maps...")
    reviews_mapped, business_df_mapped, user2idx, business2idx = build_index_maps(reviews, business_df)
    print(f"Unique users: {len(user2idx)}, unique businesses: {len(business2idx)}")

    print("Splitting interactions...")
    train, valid, test = split_interactions(reviews_mapped, test_size=test_size, val_size=val_size, random_state=random_state)
    print(f"Train/Valid/Test sizes: {len(train)}/{len(valid)}/{len(test)}")

    print("Saving processed files...")
    save_processed(train, valid, test, review_vectors, business_df_mapped, user2idx, business2idx)
    print("Done. Processed files are in 'data/processed/'.")


if __name__ == "__main__":
    run_preprocess()
