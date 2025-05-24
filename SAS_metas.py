import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import ast
# ---------- 1. Load interaction data ----------
df = pd.read_csv("all.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['user'] = user_encoder.fit_transform(df['user_id'])
df['item'] = item_encoder.fit_transform(df['item_id'])

n_users = df['user'].nunique()
n_items = df['item'].nunique()

user2items = defaultdict(list)
for row in df.itertuples():
    user2items[row.user].append(row.item)

# ---------- 2. Load and process item_meta ----------
meta = pd.read_csv("item_meta.csv")
meta = meta[meta['item_id'].isin(item_encoder.classes_)]
meta['item'] = item_encoder.transform(meta['item_id'])

# Encode category and brand
cat_encoder = LabelEncoder()
brand_encoder = LabelEncoder()
meta['category'] = cat_encoder.fit_transform(meta['main_category'].fillna('unknown'))
meta['brand'] = brand_encoder.fit_transform(meta['store'].fillna('unknown'))

# Extract information from the 'details' field
meta['detail_dict'] = meta['details'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('{') else {})
meta['item_form'] = meta['detail_dict'].apply(lambda d: d.get('Item Form', 'unknown'))
meta['skin_type'] = meta['detail_dict'].apply(lambda d: d.get('Skin Type', 'unknown'))
meta['age_range'] = meta['detail_dict'].apply(lambda d: d.get('Age Range (Description)', 'unknown'))
meta['brand_detail'] = meta['detail_dict'].apply(lambda d: d.get('Brand', 'unknown'))
meta['style'] = meta['detail_dict'].apply(lambda d: d.get('Style', 'unknown'))

# Normalize continuous features
meta['average_rating'] = pd.to_numeric(meta['average_rating'], errors='coerce').fillna(0)
meta['rating_number'] = pd.to_numeric(meta['rating_number'], errors='coerce').fillna(0)
meta['price'] = pd.to_numeric(meta['price'], errors='coerce').fillna(0)
scaler = MinMaxScaler()
meta[['average_rating', 'rating_number', 'price']] = scaler.fit_transform(meta[['average_rating', 'rating_number', 'price']])

# Encode new fields
form_encoder = LabelEncoder()
skin_encoder = LabelEncoder()
age_encoder = LabelEncoder()
brand_detail_encoder = LabelEncoder()
style_encoder = LabelEncoder()

meta['form'] = form_encoder.fit_transform(meta['item_form'])
meta['skin'] = skin_encoder.fit_transform(meta['skin_type'])
meta['age'] = age_encoder.fit_transform(meta['age_range'])
meta['brand_detail'] = brand_detail_encoder.fit_transform(meta['brand_detail'])
meta['style'] = style_encoder.fit_transform(meta['style'])

# Build mapping dictionaries
item2cat = dict(zip(meta['item'], meta['category']))
item2brand = dict(zip(meta['item'], meta['brand']))
item2feat = dict(zip(meta['item'], meta[['average_rating', 'rating_number', 'price']].values))
item2form = dict(zip(meta['item'], meta['form']))
item2skin = dict(zip(meta['item'], meta['skin']))
item2age = dict(zip(meta['item'], meta['age']))
item2brand_detail = dict(zip(meta['item'], meta['brand_detail']))
item2style = dict(zip(meta['item'], meta['style']))

n_categories = len(cat_encoder.classes_)
n_brands = len(brand_encoder.classes_)
n_forms = len(form_encoder.classes_)
n_skins = len(skin_encoder.classes_)
n_ages = len(age_encoder.classes_)
n_brand_details = len(brand_detail_encoder.classes_)
n_styles = len(style_encoder.classes_)


# ---------- 3. Build training data ----------
MAX_LEN = 50
train_data = []
for u, items in user2items.items():
    for i in range(1, len(items)):
        seq = items[:i][-MAX_LEN:]
        target = items[i]
        train_data.append((u, seq, target))

def get_meta_seq(seq, meta_map, dim=0):
    if dim == 1:
        return [meta_map.get(i, np.zeros(3)) for i in seq]
    else:
        return [meta_map.get(i, 0) for i in seq]

class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        u, seq, target = self.data[idx]
        padded_item = np.zeros(MAX_LEN, dtype=np.int64)
        padded_cat = np.zeros(MAX_LEN, dtype=np.int64)
        padded_brand = np.zeros(MAX_LEN, dtype=np.int64)
        padded_form = np.zeros(MAX_LEN, dtype=np.int64)
        padded_skin = np.zeros(MAX_LEN, dtype=np.int64)
        padded_age = np.zeros(MAX_LEN, dtype=np.int64)
        padded_brand_detail = np.zeros(MAX_LEN, dtype=np.int64)
        padded_style = np.zeros(MAX_LEN, dtype=np.int64)
        padded_feat = np.zeros((MAX_LEN, 3), dtype=np.float32)

        seq_cut = seq[-MAX_LEN:]
        padded_item[-len(seq_cut):] = seq_cut
        padded_cat[-len(seq_cut):] = get_meta_seq(seq_cut, item2cat)
        padded_brand[-len(seq_cut):] = get_meta_seq(seq_cut, item2brand)
        padded_form[-len(seq_cut):] = get_meta_seq(seq_cut, item2form)
        padded_skin[-len(seq_cut):] = get_meta_seq(seq_cut, item2skin)
        padded_age[-len(seq_cut):] = get_meta_seq(seq_cut, item2age)
        padded_brand_detail[-len(seq_cut):] = get_meta_seq(seq_cut, item2brand_detail)
        padded_style[-len(seq_cut):] = get_meta_seq(seq_cut, item2style)
        padded_feat[-len(seq_cut):] = get_meta_seq(seq_cut, item2feat, dim=1)

        return (
            torch.tensor(padded_item),
            torch.tensor(padded_cat),
            torch.tensor(padded_brand),
            torch.tensor(padded_form),
            torch.tensor(padded_skin),
            torch.tensor(padded_age),
            torch.tensor(padded_brand_detail),
            torch.tensor(padded_style),
            torch.tensor(padded_feat),
            torch.tensor(target)
        )


# ---------- 4. Define SASRec model with meta ----------
class SASRecWithMeta(nn.Module):
    def __init__(self, n_items, n_token_dict, hidden_units=64, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, hidden_units, padding_idx=0)
        self.meta_embs = nn.ModuleDict({
            k: nn.Embedding(v, hidden_units, padding_idx=0) for k, v in n_token_dict.items()
        })
        self.feat_proj = nn.Linear(3, hidden_units)
        self.pos_emb = nn.Embedding(max_len, hidden_units)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=4, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(0.2)

    def forward(self, item_seq, *meta_seqs, feat_seq):
        x = self.item_emb(item_seq)
        for key, meta_seq in zip(self.meta_embs.keys(), meta_seqs):
            x += self.meta_embs[key](meta_seq)
        x += self.feat_proj(feat_seq)
        x += self.pos_emb(torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0))
        x = self.dropout(self.norm(x))
        padding_mask = (item_seq == 0)
        x = self.transformer(x.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)
        return x

    def predict(self, item_seq, *meta_seqs, feat_seq):
        x = self.forward(item_seq, *meta_seqs, feat_seq=feat_seq)
        return torch.matmul(x[:, -1, :], self.item_emb.weight.T)

        
def generate_submission_and_evaluate(
    model,
    all_df,                # all.csv
    submission_df,         # sample_submission.csv
    user_encoder,
    item_encoder,
    user2items,
    max_len=50,
    output_file='submission_metas.csv',
    device='cpu'
):
    model.eval()

    # Decode and store user mapping (ensure consistency)
    submission_df['user'] = user_encoder.transform(submission_df['user_id'])

    # === Generate recommendation list ===
    recommendations = []
    for row in submission_df.itertuples():
        user = row.user
        seq = user2items.get(user, [])[-max_len:]
        seq_cat = get_meta_seq(seq, item2cat)
        seq_brand = get_meta_seq(seq, item2brand)
        seq_feat = get_meta_seq(seq, item2feat, dim=1)

        padded_item = np.zeros(max_len, dtype=np.int64)
        padded_cat = np.zeros(max_len, dtype=np.int64)
        padded_brand = np.zeros(max_len, dtype=np.int64)
        padded_feat = np.zeros((max_len, 3), dtype=np.float32)

        padded_item[-len(seq):] = seq
        padded_cat[-len(seq):] = seq_cat
        padded_brand[-len(seq):] = seq_brand
        padded_feat[-len(seq):] = seq_feat

        item_tensor = torch.tensor(np.array([padded_item]), dtype=torch.long).to(device)
        cat_tensor = torch.tensor(np.array([padded_cat]), dtype=torch.long).to(device)
        brand_tensor = torch.tensor(np.array([padded_brand]), dtype=torch.long).to(device)
        feat_tensor = torch.tensor(np.array([padded_feat]), dtype=torch.float32).to(device)

        with torch.no_grad():
            scores = model.predict(item_tensor, cat_tensor, brand_tensor, feat_seq=feat_tensor).cpu().numpy()[0]

        seen = set(user2items.get(user, []))
        scores[list(seen)] = -np.inf
        top_items = np.argsort(scores)[-10:][::-1]
        top_item_ids = item_encoder.inverse_transform(top_items)
        recommendations.append(','.join(map(str, top_item_ids)))

    submission_df['item_id'] = recommendations
    submission_df['ID'] = submission_df['user_id']
    submission_df = submission_df[['ID', 'user_id', 'item_id']]
    submission_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"✅ recommendation is saved to {output_file}")

    # === Recall@10 Evaluation ===
    all_df = all_df.sort_values(by='timestamp')
    all_df['user'] = user_encoder.transform(all_df['user_id'])
    all_df['item'] = item_encoder.transform(all_df['item_id'])

    user_last_item = all_df.groupby('user')['item'].agg(lambda x: x.iloc[-1]).to_dict()

    hit = 0
    total = 0
    for row in submission_df.itertuples():
        user = user_encoder.transform([row.user_id])[0]
        if user not in user_last_item:
            continue
        true_item = user_last_item[user]
        predicted_items = set(map(int, row.item_id.split(',')))
        if true_item in predicted_items:
            hit += 1
        total += 1

    recall = hit / total if total > 0 else 0.0
    print(f"🎯 Recall@10: {recall:.4f} ({hit}/{total} users hit)")
    return recall

# ------------------ 4. Train model ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_token_dict = {
    'cat': n_categories,
    'brand': n_brands,
    'form': n_forms,
    'skin': n_skins,
    'age': n_ages,
    'brand_detail': n_brand_details,
    'style': n_styles,
}

model = SASRecWithMeta(n_items=n_items, n_token_dict=n_token_dict, hidden_units=64, max_len=MAX_LEN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------- 6. Training ----------
dataset = SeqDataset(train_data)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        item_seq, *meta_seqs, feat_seq, targets = (x.to(device) for x in batch)
        logits = model.predict(item_seq, *meta_seqs, feat_seq=feat_seq)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} | Loss: {total_loss:.4f}")

# ------------------ 5. Inference and Recommendation ------------------
model.eval()
submission = pd.read_csv('sample_submission.csv')
submission['user'] = user_encoder.transform(submission['user_id'])

recommendations = []
for row in submission.itertuples():
    user = row.user
    seq = user2items[user][-MAX_LEN:]
    seq_cat = get_meta_seq(seq, item2cat)
    seq_brand = get_meta_seq(seq, item2brand)
    seq_feat = get_meta_seq(seq, item2feat, dim=1)

    padded_item = np.zeros(MAX_LEN, dtype=np.int64)
    padded_cat = np.zeros(MAX_LEN, dtype=np.int64)
    padded_brand = np.zeros(MAX_LEN, dtype=np.int64)
    padded_feat = np.zeros((MAX_LEN, 3), dtype=np.float32)

    padded_item[-len(seq):] = seq
    padded_cat[-len(seq):] = seq_cat
    padded_brand[-len(seq):] = seq_brand
    padded_feat[-len(seq):] = seq_feat

    item_tensor = torch.tensor(np.array([padded_item]), dtype=torch.long).to(device)
    cat_tensor = torch.tensor(np.array([padded_cat]), dtype=torch.long).to(device)
    brand_tensor = torch.tensor(np.array([padded_brand]), dtype=torch.long).to(device)
    feat_tensor = torch.tensor(np.array([padded_feat]), dtype=torch.float32).to(device)

    with torch.no_grad():
        scores = model.predict(item_tensor, cat_tensor, brand_tensor, feat_seq=feat_tensor).cpu().numpy()[0]

    seen = set(user2items[user])
    scores[list(seen)] = -np.inf  # Avoid recommending items already interacted with
    top_items = np.argsort(scores)[-10:][::-1]
    top_item_ids = item_encoder.inverse_transform(top_items)
    recommendations.append(','.join(map(str, top_item_ids)))

submission['item_id'] = recommendations
submission['ID'] = submission['user_id']
submission = submission[['ID', 'user_id', 'item_id']]
submission.to_csv('submission_metas.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
print("✅ recommendation is saved to submission_metas.csv")

submission_df = pd.read_csv('sample_submission.csv')

# 生成并评估
generate_submission_and_evaluate(
    model=model,
    all_df=df,
    submission_df=submission_df,
    user_encoder=user_encoder,
    item_encoder=item_encoder,
    user2items=user2items,
    max_len=MAX_LEN,
    output_file='submission_metas.csv',
    device=device
)