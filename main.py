from src.model import ReviewAwareRec
from src.data_preprocess import run_preprocess
from pathlib import Path
import json
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import ReviewDataset
from src.train import train_model

PROC_DIR = Path("data/processed")

def main(preprocess=True, version="v1"):
    if preprocess:
        run_preprocess()

    # 加载数据
    train_csv = PROC_DIR / "train.csv"
    valid_csv = PROC_DIR / "valid.csv"
    test_csv = PROC_DIR / "test.csv"  # 需确保有 test.csv

    meta = json.load(open(PROC_DIR / "meta.json"))
    review_vecs = np.load(PROC_DIR / "review_vectors.npy")
    business_feats = np.load(PROC_DIR / "business_features.npy")

    # DataLoader
    train_ds = ReviewDataset(train_csv, review_vecs, business_feats)
    valid_ds = ReviewDataset(valid_csv, review_vecs, business_feats)
    test_ds = ReviewDataset(test_csv, review_vecs, business_feats)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    # 训练 + 测试
    train_model(train_loader, valid_loader, meta, test_loader=test_loader, n_epochs=10, lr=1e-3, version=version)


if __name__ == "__main__":
    main(preprocess=False, version="v1")
