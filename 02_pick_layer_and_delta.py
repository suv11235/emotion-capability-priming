import argparse
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
args = p.parse_args()

d = torch.load(f"{args.out}/directions.pt", weights_only=False)["directions"].numpy()
val = np.load(f"{args.out}/activations_val.npz")

def auc_at(L):
    pos, neg = val["sad"][:, L] @ d[L], val["other"][:, L] @ d[L]
    return roc_auc_score([1] * len(pos) + [0] * len(neg), np.r_[pos, neg])

n_layers = d.shape[0]
layer = max(range(int(0.1 * n_layers), n_layers), key=auc_at)

delta = float((val["sad"][:, layer] @ d[layer]).mean() - (val["other"][:, layer] @ d[layer]).mean())
json.dump({"layer": int(layer), "delta": delta}, open(f"{args.out}/frozen.json", "w"), indent=2)
print(f"layer {layer}, delta {delta:.4f}")
