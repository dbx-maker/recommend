import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from pathlib import Path
from datetime import datetime

from src.model import ReviewAwareRec

def evaluate_model(model, data_loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for user_idx, business_idx, review_vec, business_vec, rating in data_loader:
            user_idx, business_idx = user_idx.to(device), business_idx.to(device)
            review_vec, business_vec = review_vec.to(device), business_vec.to(device)
            rating = rating.to(device)

            pred = model(user_idx, business_idx, review_vec, business_vec)
            preds.append(pred.cpu().numpy())
            trues.append(rating.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    return mse, rmse, mae, r2, preds, trues


def train_model(train_loader, valid_loader, meta, test_loader=None, n_epochs=5, lr=1e-4, device=None, version="v1"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReviewAwareRec(
        n_users=meta["n_users"],
        n_businesses=meta["n_businesses"],
        review_dim=768,
        business_dim=21,
        embed_dim=32
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for user_idx, business_idx, review_vec, business_vec, rating in tqdm(train_loader, desc=f"Epoch {epoch}"):
            user_idx, business_idx = user_idx.to(device), business_idx.to(device)
            review_vec, business_vec = review_vec.to(device), business_vec.to(device)
            rating = rating.to(device)

            pred = model(user_idx, business_idx, review_vec, business_vec)
            loss = criterion(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(rating)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # 验证集
        mse, rmse, mae, r2, _, _ = evaluate_model(model, valid_loader, device)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Valid MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # 保存模型
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"review_rec_{version}_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # 如果有测试集，评估测试集性能
    if test_loader is not None:
        mse, rmse, mae, r2, preds, trues = evaluate_model(model, test_loader, device)
        print("\n📊 Test Set Evaluation:")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")

        # 保存预测结果
        out_csv = model_dir / f"test_predictions_{version}_{timestamp}.csv"
        np.savetxt(out_csv, np.vstack([trues, preds]).T, delimiter=",", header="true,pred", comments="")
        print(f"📝 Test predictions saved to {out_csv}")

    return model

