import random, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from tqdm import tqdm
import os

# 1) Fix seeds
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# 2) Load interactions and build user↔item graph
def load_interactions(path):
    df = pd.read_csv(path)
    df['label']=1
    return df

# 3) Generate and save frozen BERT item vectors
def gen_item_bert(meta_path, item2id, out="item_bert.npy"):
    tok=BertTokenizer.from_pretrained('bert-base-uncased')
    mdl=BertModel.from_pretrained('bert-base-uncased').eval()
    V=len(item2id); D=768; M=np.zeros((V,D))
    df=pd.read_csv(meta_path)
    for _,r in tqdm(df.iterrows(),total=len(df)):
        if r['item_id'] not in item2id: continue
        idx=item2id[r['item_id']]
        txt=f"{r['title']} {r.get('category','')}"
        toks=tok(txt,return_tensors='pt',truncation=True,padding=True)
        with torch.no_grad():
            emb=mdl(**toks).last_hidden_state.mean(1).squeeze().numpy()
        M[idx]=emb
    np.save(out,M); print("Saved item BERT→",out)

# 4) LightGCN dataset
class LightGCNDataset(Dataset):
    def __init__(self, interactions, num_users, num_items, neg=1):
        self.pos = interactions.values.tolist()
        self.U, self.I = num_users, num_items
        self.neg = neg
        self.user_hist = interactions.groupby('user_id')['item_id'].apply(set).to_dict()
    def __len__(self): return len(self.pos)
    def __getitem__(self, i):
        u,i,_=self.pos[i]
        negs = random.sample(range(self.I), self.neg)
        return u, i, negs[0]

# 5) LightGCN model (K layers)
class LightGCN(nn.Module):
    def __init__(self, U, I, emb_dim=64, K=3):
        super().__init__()
        self.U, self.I = U,I
        self.k = K
        self.user_emb = nn.Embedding(U,emb_dim)
        self.item_emb = nn.Embedding(I,emb_dim)
    def forward(self, adj):
        # ...existing code to propagate embeddings K steps...
        return self.user_emb.weight, self.item_emb.weight
    
    def bpr_loss(self, u,i,j):
        eu=self.user_emb(u); 
        vi=self.item_emb(i); vj=self.item_emb(j)
        pos=(eu*vi).sum(1); neg=(eu*vj).sum(1)
        return -torch.log(torch.sigmoid(pos-neg)).mean()

# 6) NeuMF head
class NeuMFHead(nn.Module):
    def __init__(self, u_emb, i_bert, mlp_dims=[128,64]):
        super().__init__()
        self.u_dim = u_emb.shape[1]
        self.i_dim = i_bert.shape[1]
        self.u_mat = nn.Parameter(torch.tensor(u_emb),requires_grad=False)
        self.i_mat = nn.Parameter(torch.tensor(i_bert),requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.u_dim+self.i_dim, mlp_dims[0]), nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]), nn.ReLU(),
            nn.Linear(mlp_dims[1],1)
        )
    def forward(self, u,i):
        ue = self.u_mat[u]; ie = self.i_mat[i]
        return self.mlp(torch.cat([ue,ie],-1)).squeeze(-1)

# 7) Training pipelines
def train_lightgcn(df, U,I, epochs=10, lr=0.01):
    # build adjacency
    edges = df[['user_id','item_id']].values
    # ...construct adjacency matrix or sparse graph...
    ds=LightGCNDataset(df[['user_id','item_id','label']],U,I)
    dl=DataLoader(ds,batch_size=1024,shuffle=True)
    model=LightGCN(U,I); opt=optim.Adam(model.parameters(),lr=lr)
    for e in range(epochs):
        tot=0
        for u,i,j in dl:
            loss=model.bpr_loss(u,i,j)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"LightGCN E{e+1} Loss={tot/len(dl):.4f}")
    return model.user_emb.weight.detach().cpu().numpy()

def train_neumf(head, train_df, device, epochs=5, bs=512):
    ds = LightGCNDataset(train_df[['user_id','item_id','label']], 
                         head.u_mat.shape[0], head.i_mat.shape[0], neg=1)
    dl = DataLoader(ds, bs, shuffle=True)
    head.to(device)
    opt=optim.Adam(head.parameters(),lr=1e-3); crit=nn.BCEWithLogitsLoss()
    for e in range(epochs):
        tot=0
        for u,i,j in dl:
            # here j is a negative sample index
            pos_pred = head(u.to(device), i.to(device))
            neg_pred = head(u.to(device), j.to(device))
            lbl = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
            pred = torch.cat([pos_pred, neg_pred])
            loss=crit(pred, lbl.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"NeuMF E{e+1} Loss={tot/len(dl):.4f}")

# 8) Eval with negative sampling
def eval_rec(head, full_df, test_df, top_k=10):
    hist = full_df.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items=set(range(head.i_mat.size(0)))
    recalls=[]; hits=[]
    for u, group in test_df.groupby("user_id"):
        gt=set(group["item_id"])
        cand=list(all_items-hist.get(u,set()))
        scores=head(torch.tensor([u]*len(cand)), torch.tensor(cand))
        idx=torch.topk(scores,top_k).indices.tolist()
        top=[cand[i] for i in idx]
        h=len(set(top)&gt)
        recalls.append(h/len(gt)); hits.append(1 if h>0 else 0)
    print(f"Recall@{top_k}={np.mean(recalls):.4f}, Hit@{top_k}={np.mean(hits):.4f}")

if __name__=="__main__":
    set_seed(1000)
    # ...load train/test/meta files...
    train=load_interactions("train.csv")
    test=pd.read_csv("test.csv")
    # map users/items to 0..N
    us, iset = train.user_id.unique(), train.item_id.unique()
    u2i={u:i for i,u in enumerate(us)}; i2i={i:j for j,i in enumerate(iset)}
    train['user_id']=train['user_id'].map(u2i); train['item_id']=train['item_id'].map(i2i)
    test['user_id']=test['user_id'].map(u2i); test['item_id']=test['item_id'].map(i2i)

    # ensure frozen BERT vectors exist
    if not os.path.isfile("item_bert.npy"):
        gen_item_bert("item_meta.csv", i2i, out="item_bert.npy")
    item_bert = np.load("item_bert.npy")

    user_emb = train_lightgcn(train, len(us), len(iset))
    head = NeuMFHead(user_emb, item_bert)
    train_neumf(head, train, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    eval_rec(head, train, test)
