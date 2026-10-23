import argparse
import json
import os
import numpy as np
import pandas as pd
from datasets import load_dataset

from common import SEED, auc, build_direction, load_model, story_activations

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
A = {name: story_activations(model, processor, df.story.tolist()) for name, df in
     dict(sad_tr=sad_tr, sad_va=sad_va, sad_te=sad_te, oth_tr=oth_tr,
          oth_va=oth_va, oth_te=oth_te, neu_tr=neu_tr).items()}

directions = build_direction(A["sad_tr"], A["oth_tr"], A["neu_tr"], args.pca)

rows = []
for split_name in ["va", "te"]:
    for L in range(directions.shape[0]):
        rows.append(dict(split=split_name, layer=L,
                         auc=auc(A[f"sad_{split_name}"][:, L], A[f"oth_{split_name}"][:, L], directions[L])))
pd.DataFrame(rows).to_csv(f"{args.out}/auc_by_layer.csv", index=False)

import torch
torch.save({"directions": torch.from_numpy(directions), "emotion": args.emotion}, f"{args.out}/directions.pt")
np.savez(f"{args.out}/activations_val.npz", sad=A["sad_va"], other=A["oth_va"])

np.savez(f"{args.out}/activations_train_test.npz", sad_tr=A["sad_tr"], oth_tr=A["oth_tr"],
         neu_tr=A["neu_tr"], sad_te=A["sad_te"], oth_te=A["oth_te"])

heldout = sorted(int(i) for df in (sad_va, sad_te, oth_va, oth_te) for i in df.index)
json.dump({"stories_parquet_rows": heldout}, open(f"{args.out}/heldout_rows.json", "w"))
print(pd.DataFrame(rows).query("split=='te'").sort_values("auc").tail(3))
