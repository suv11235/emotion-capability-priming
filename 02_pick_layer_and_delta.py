"""Step 2: choose the layer and the size of one "push" (delta).

Reads : out/directions.pt, out/activations_val.npz  (VALIDATION stories only)
Writes: out/frozen.json   -> {"layer": 10, "delta": 1.52}

Rules I set before running:
  * layer = best validation AUC (skipping the first 10% of layers)
  * delta = how far apart sad and non-sad stories sit along the arrow
  * both are chosen on validation data and then frozen. Test data is never used.
"""
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

# delta = mean(sad projection) - mean(other projection)
delta = float((val["sad"][:, layer] @ d[layer]).mean() - (val["other"][:, layer] @ d[layer]).mean())
json.dump({"layer": int(layer), "delta": delta}, open(f"{args.out}/frozen.json", "w"), indent=2)
print(f"layer {layer}, delta {delta:.4f}")
