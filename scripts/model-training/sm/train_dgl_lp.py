# train_dgl_lp.py
import os, glob, io, argparse
import pandas as pd
import boto3
import torch
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv

def load_csv_local(path):
    return pd.read_csv(path)

def build_id_map(series):
    uniq = series.astype(str).dropna().unique().tolist()
    return {u: i for i, u in enumerate(uniq)}

class LinkPred(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid, "mean")
        self.conv2 = SAGEConv(hid, out_dim, "mean")
    def forward(self, g, x):
        h = self.conv1(g, x).relu()
        h = self.conv2(g, h)
        return h

def neg_sample(g, k=4):
    src, _ = g.edges()
    N = g.num_nodes()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, N, (src.numel()*k,), device=src.device)
    return neg_src, neg_dst

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--neg", type=int, default=4)
    args = ap.parse_args()

    # 1) SageMaker provides the TRAIN channel path via env var:
    #    e.g., /opt/ml/input/data/train
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_dir or not os.path.isdir(train_dir):
        raise RuntimeError(f"TRAIN channel not found: {train_dir}")

    # Expect the three processed CSVs to be present in the train channel
    v_hcp_path = os.path.join(train_dir, "vertices_hcp.csv")
    v_tac_path = os.path.join(train_dir, "vertices_tactic.csv")
    e_eng_path = os.path.join(train_dir, "edges_engaged.csv")
    for p in (v_hcp_path, v_tac_path, e_eng_path):
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing required file in TRAIN channel: {p}")

    # 2) Load CSVs (Gremlin-CSV headers from Neptune export/prepare step)
    v_hcp = load_csv_local(v_hcp_path)        # needs columns: ~id, ~label, ...
    v_tac = load_csv_local(v_tac_path)        # needs columns: ~id, ~label, ...
    e_eng = load_csv_local(e_eng_path)        # needs columns: ~from, ~to, ~label, ...

    # 3) Map IDs to integers
    hcp_map = build_id_map(v_hcp["~id"])
    tac_map = build_id_map(v_tac["~id"])

    src = e_eng["~from"].astype(str).map(hcp_map).dropna().astype("int64").values
    dst = e_eng["~to"].astype(str).map(tac_map).dropna().astype("int64").values
    src = torch.tensor(src, dtype=torch.int64)
    dst = torch.tensor(dst, dtype=torch.int64)

    # 4) Build heterograph and minimal LP baseline
    hg = dgl.heterograph({("HCP","ENGAGED","Tactic"): (src, dst)},
                         num_nodes_dict={"HCP": len(hcp_map), "Tactic": len(tac_map)})

    emb_dim = 64
    hcp_emb = torch.nn.Embedding(hg.num_nodes("HCP"), emb_dim)
    tac_emb = torch.nn.Embedding(hg.num_nodes("Tactic"), emb_dim)

    g = dgl.to_homogeneous(hg)
    
    model = LinkPred(in_dim=emb_dim, hid=args.hidden, out_dim=64)
    
    # Create optimizer with embedding parameters properly
    opt = torch.optim.Adam(list(model.parameters()) + list(hcp_emb.parameters()) + list(tac_emb.parameters()), lr=1e-3)

    for ep in range(args.epochs):
        model.train()
        
        # Get embeddings and concatenate them properly
        x = torch.cat([hcp_emb.weight, tac_emb.weight], dim=0)
        
        h = model(g, x)
        pos_src, pos_dst = g.edges()
        pos_score = (h[pos_src] * h[pos_dst]).sum(dim=1)
        neg_src, neg_dst = neg_sample(g, args.neg)
        neg_score = (h[neg_src] * h[neg_dst]).sum(dim=1)
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch={ep} loss={loss.item():.4f}")

    # 5) Save artifacts to the SageMaker model directory
    out_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"hcp_emb": hcp_emb.state_dict(),
                "tactic_emb": tac_emb.state_dict()},
               os.path.join(out_dir, "node_emb.pt"))
