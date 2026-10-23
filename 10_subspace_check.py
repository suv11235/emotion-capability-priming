import argparse
import json
import numpy as np
import torch
from sklearn.decomposition import PCA

from common import SEED

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out", help="step 1's folder")
ap.add_argument("--directions", nargs="+", required=True)
ap.add_argument("--layers", default=None, help="default: the frozen layer")
ap.add_argument("--variance", type=float, default=0.95)
ap.add_argument("--n_random", type=int, default=2)
a = ap.parse_args()

neutral = np.load(f"{a.out}/activations_train_test.npz")["neu_tr"]
layers = [int(x) for x in a.layers.split(",")] if a.layers else [json.load(open(f"{a.out}/frozen.json"))["layer"]]
dirs = {path: torch.load(path, weights_only=False) for path in a.directions}
dims = neutral.shape[2]
rand = np.random.default_rng(SEED).standard_normal((a.n_random, dims))

for L in layers:
    pca = PCA().fit(neutral[:, L].astype(np.float32))
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), a.variance) + 1)
    span = pca.components_[:k]

    def inside(v):
        return float(np.linalg.norm(span @ (v / np.linalg.norm(v))))

    parts = [f"{d.get('emotion', p)} {inside(d['directions'][L].numpy()):.2f}" for p, d in dirs.items()]
    parts += [f"random_{i} {inside(r):.2f}" for i, r in enumerate(rand)]
    print(f"layer {L:2d}: {k} of {dims} dims hold {a.variance:.0%} of the variance | "
          f"fraction of length inside: {', '.join(parts)} | chance {np.sqrt(k / dims):.2f}")
