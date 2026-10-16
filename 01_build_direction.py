import argparse
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from common import SEED, load_model

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--emotion", default="sad")
p.add_argument("--n_other", type=int, default=1200)
p.add_argument("--pca", type=float, default=0.50)
p.add_argument("--out", default="out")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(SEED)

base = "expression/"
stories = load_dataset("ryancodrai/emotion-probes", data_files=base + "stories.parquet", split="train").to_pandas()
neutral = load_dataset("ryancodrai/emotion-probes", data_files=base + "neutral_stories.parquet", split="train").to_pandas()
sad = stories[stories.emotion.str.lower() == args.emotion]
other = stories[stories.emotion.str.lower() != args.emotion]

other = other.groupby("topic", group_keys=False).apply(
    lambda g: g.sample(min(len(g), args.n_other // 100), random_state=SEED))

def split(df):
    idx = rng.permutation(len(df))
    a, b = int(0.6 * len(df)), int(0.8 * len(df))
    return df.iloc[idx[:a]], df.iloc[idx[a:b]], df.iloc[idx[b:]]

sad_tr, sad_va, sad_te = split(sad)
oth_tr, oth_va, oth_te = split(other)
neu_tr, _, _ = split(neutral)

model, processor = load_model(args.model)

def activations(texts, start_token=50):
    import torch
    out = []
    for t in texts:
        inp = processor(text=t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            hs = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
        n = inp["input_ids"].shape[1]
        s = min(start_token, n // 2)
        out.append(torch.stack([h[0, s:n].float().mean(0).cpu() for h in hs]).numpy())
    return np.stack(out)

A = {name: activations(df.story.tolist()) for name, df in
     dict(sad_tr=sad_tr, sad_va=sad_va, sad_te=sad_te, oth_tr=oth_tr,
          oth_va=oth_va, oth_te=oth_te, neu_tr=neu_tr).items()}

raw = A["sad_tr"].mean(0) - A["oth_tr"].mean(0)
directions = np.zeros_like(raw)
for L in range(raw.shape[0]):
    v = raw[L].copy()
    if args.pca > 0:

        pca = PCA().fit(A["neu_tr"][:, L])
        k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), args.pca) + 1)
        pcs = pca.components_[:k]
        v = v - pcs.T @ (pcs @ v)
    directions[L] = v / (np.linalg.norm(v) + 1e-8)

rows = []
for split_name in ["va", "te"]:
    for L in range(directions.shape[0]):
        pos = A[f"sad_{split_name}"][:, L] @ directions[L]
        neg = A[f"oth_{split_name}"][:, L] @ directions[L]
        auc = roc_auc_score([1] * len(pos) + [0] * len(neg), np.r_[pos, neg])
        rows.append(dict(split=split_name, layer=L, auc=auc))
pd.DataFrame(rows).to_csv(f"{args.out}/auc_by_layer.csv", index=False)

import torch
torch.save({"directions": torch.from_numpy(directions), "emotion": args.emotion}, f"{args.out}/directions.pt")
np.savez(f"{args.out}/activations_val.npz", sad=A["sad_va"], other=A["oth_va"])
print(pd.DataFrame(rows).query("split=='te'").sort_values("auc").tail(3))
