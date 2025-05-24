import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import random
import time
import os
# Recall@10 0.0029
# ---------- Utility ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------- Data Loading ----------
def load_data_with_negatives(filename, neg_train=3, neg_valid=10, train_ratio=0.85):
    df = pd.read_csv(filename)
    df["label"] = 1

    all_users = df["user_id"].unique()
    all_items = df["item_id"].unique()
    user2id = {u: i for i, u in enumerate(all_users)}
    item2id = {i: j for j, i in enumerate(all_items)}
    id2item = {v: k for k, v in item2id.items()}

    df["user_id"] = df["user_id"].map(user2id)
    df["item_id"] = df["item_id"].map(item2id)

    grouped = df.groupby("user_id")["item_id"].apply(list)
    all_item_set = set(item2id.values())

    train_rows, val_rows = [], []

    for user_id, items in grouped.items():
        if len(items) < 2: continue
        items = np.array(items)
        np.random.shuffle(items)
        split = int(len(items) * train_ratio)
        train_pos, val_pos = items[:split], items[split:]
        interacted = set(items)
        neg_pool = list(all_item_set - interacted)

        if len(neg_pool) < len(train_pos) * neg_train:
            continue

        train_neg = random.sample(neg_pool, len(train_pos) * neg_train)
        val_neg = random.sample(neg_pool, neg_valid)

        train_rows.extend([(user_id, i, 1) for i in train_pos])
        train_rows.extend([(user_id, i, 0) for i in train_neg])
        val_rows.extend([(user_id, i, 1) for i in val_pos])
        val_rows.extend([(user_id, i, 0) for i in val_neg])

    train_df = pd.DataFrame(train_rows, columns=["user_id", "item_id", "label"])
    val_df = pd.DataFrame(val_rows, columns=["user_id", "item_id", "label"])
    return train_df, val_df, len(user2id), len(item2id), user2id, item2id, id2item, df

# ---------- Model ----------
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=128, mlp_dims=[256, 128]):
        super().__init__()
        self.user_embedding_gmf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, mf_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_dims[0] // 2)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_dims[0] // 2)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(mf_dim + mlp_dims[1], 1)

    def forward(self, user_idx, item_idx):
        gmf = self.user_embedding_gmf(user_idx) * self.item_embedding_gmf(item_idx)
        mlp_input = torch.cat([
            self.user_embedding_mlp(user_idx),
            self.item_embedding_mlp(item_idx)
        ], dim=-1)
        mlp_out = self.mlp(mlp_input)
        out = torch.cat([gmf, mlp_out], dim=-1)
        return self.output(out)

# ---------- Training ----------
def train_model(model, train_loader, val_loader, device, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    best_loss = float("inf")
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, l in train_loader:
            u, i, l = u.to(device), i.to(device), l.to(device)
            optimizer.zero_grad()
            pred = model(u, i).squeeze(-1)
            loss = criterion(pred, l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for u, i, l in val_loader:
                u, i, l = u.to(device), i.to(device), l.to(device)
                pred = model(u, i).squeeze(-1)
                val_loss += criterion(pred, l).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model

# ---------- Evaluation ----------
def evaluate_recall_hit(model, full_df, test_df, item2id, device, top_k=10):
    all_items = set(item2id.values())
    test_df = test_df[test_df["user_id"].isin(user2id)]
    test_df["user_id"] = test_df["user_id"].map(user2id)
    test_df["item_id"] = test_df["item_id"].map(item2id)

    user_history = full_df.groupby("user_id")["item_id"].apply(set).to_dict()
    test_gt = defaultdict(set)
    for _, row in test_df.iterrows():
        test_gt[row["user_id"]].add(row["item_id"])

    recall_list, hit_list = [], []

    model.eval()
    with torch.no_grad():
        for user_id in test_gt:
            seen = user_history.get(user_id, set())
            candidate_items = list(all_items - seen)

            user_tensor = torch.full((len(candidate_items),), user_id, dtype=torch.long).to(device)
            item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)

            scores = torch.sigmoid(model(user_tensor, item_tensor).squeeze(-1))
            top_indices = torch.topk(scores, k=top_k).indices.cpu().numpy()
            top_items = [candidate_items[i] for i in top_indices]

            hits = len(set(top_items) & test_gt[user_id])
            recall = hits / len(test_gt[user_id])
            recall_list.append(recall)
            hit_list.append(1 if hits > 0 else 0)

    print(f"Recall@{top_k}: {np.mean(recall_list):.4f}, Hit@{top_k}: {np.mean(hit_list):.4f}")


# ---------- Main ----------
if __name__ == "__main__":
    set_seed(1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = "train.csv"
    test_file = "test.csv"
    # submit_file = "sample_submission.csv"

    train_df, val_df, num_users, num_items, user2id, item2id, id2item, full_df = load_data_with_negatives(train_file)

    train_dataset = TensorDataset(
        torch.tensor(train_df["user_id"].values),
        torch.tensor(train_df["item_id"].values),
        torch.tensor(train_df["label"].values, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_df["user_id"].values),
        torch.tensor(val_df["item_id"].values),
        torch.tensor(val_df["label"].values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    model = NeuMF(num_users, num_items).to(device)
    model = train_model(model, train_loader, val_loader, device)

    evaluate_recall_hit(model, full_df, pd.read_csv(test_file), id2item, item2id, device)
