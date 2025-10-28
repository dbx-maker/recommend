import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, csv_path, review_vectors, business_features):
        """
        Args:
            csv_path: train/valid/test CSV
            review_vectors: np.ndarray, shape=(n_reviews, 768)
            business_features: np.ndarray, shape=(n_businesses, 21)
        """
        self.df = pd.read_csv(csv_path)
        self.review_vectors = review_vectors  # ndarray
        self.business_features = business_features  # ndarray

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_idx = torch.tensor(row["user_idx"], dtype=torch.long)
        business_idx = torch.tensor(row["business_idx"], dtype=torch.long)
        rating = torch.tensor(row["stars"], dtype=torch.float32)

        # review_vec 使用 review_index
        review_vec = torch.tensor(self.review_vectors[int(row["review_index"])], dtype=torch.float32)

        # business_vec 使用 business_idx
        business_vec = torch.tensor(self.business_features[int(row["business_idx"])], dtype=torch.float32)

        return user_idx, business_idx, review_vec, business_vec, rating
