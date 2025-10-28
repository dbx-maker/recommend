# src/model.py
import torch
import torch.nn as nn

class ReviewAwareRec(nn.Module):
    def __init__(self, n_users, n_businesses, review_dim=768, business_dim=21, embed_dim=32):
        super().__init__()
        # Embedding 部分（协同过滤）
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.business_emb = nn.Embedding(n_businesses, embed_dim)

        # review + business 特征部分（内容过滤）
        self.review_fc = nn.Linear(review_dim, 128)
        self.business_fc = nn.Linear(business_dim, embed_dim)

        # 最终 MLP
        self.mlp = nn.Sequential(
            nn.Linear(128 + embed_dim * 2 + 21, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user_idx, business_idx, review_vec, business_vec):
        u = self.user_emb(user_idx)
        b = self.business_emb(business_idx)
        r = torch.relu(self.review_fc(review_vec))
        #bf = torch.relu(self.business_fc(business_vec))

        # 拼接四种信息
        x = torch.cat([u, b, r, business_vec], dim=1)
        return self.mlp(x).squeeze(-1)
