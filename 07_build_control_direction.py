import argparse
import json
import os
import numpy as np
import torch

from common import SEED, auc, build_direction

p = argparse.ArgumentParser()
mode = p.add_mutually_exclusive_group(required=True)
mode.add_argument("--shuffle", action="store_true")
mode.add_argument("--topic", default=None)
p.add_argument("--shuffle_seed", type=int, default=0)
p.add_argument("--model", default=None, help="needed for --topic")
p.add_argument("--out", default="out", help="step 1's folder")
p.add_argument("--out_dir", required=True)
p.add_argument("--n", type=int, default=1200, help="stories per class for --topic")
p.add_argument("--pca", type=float, default=0.50)
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

A = np.load(f"{args.out}/activations_train_test.npz")
layer = json.load(open(f"{args.out}/frozen.json"))["layer"]
sad_dir = torch.load(f"{args.out}/directions.pt", weights_only=False)["directions"].numpy()
info = {}

if args.shuffle:

    pos, neg = A["sad_tr"], A["oth_tr"]
    rng = np.random.default_rng(args.shuffle_seed)
    n_pos, n_neg = len(pos), len(neg)
    k = round(n_pos * n_pos / (n_pos + n_neg))
    pp, nn = rng.permutation(n_pos), n_pos + rng.permutation(n_neg)
    pooled = np.concatenate([pos, neg])
    new_pos = pooled[np.sort(np.r_[pp[:k], nn[:n_pos - k]])]
    new_neg = pooled[np.sort(np.r_[pp[k:], nn[n_pos - k:]])]
    directions = build_direction(new_pos, new_neg, A["neu_tr"], args.pca)
    name = "shuffled" if args.shuffle_seed == 0 else f"shuffled_s{args.shuffle_seed}"
    info.update(kind="shuffled_labels", shuffle_seed=args.shuffle_seed)
else:
    import pandas as pd
    from datasets import load_dataset
    from common import load_model, story_activations
    stories = load_dataset("ryancodrai/emotion-probes", data_files="expression/stories.parquet",
                           split="train").to_pandas()
    heldout = set(json.load(open(f"{args.out}/heldout_rows.json"))["stories_parquet_rows"])
    eligible = stories[~stories.index.isin(heldout)]
    counts = eligible.topic.value_counts()
    topic = args.topic if args.topic != "auto" else min(counts.index, key=lambda t: (-counts[t], t))
    if topic not in counts.index:
        raise SystemExit(f"--topic {topic!r} is not a topic with eligible stories")

    def balanced(df, by, n):

        quota = max(1, n // df[by].nunique())
        return df.groupby(by, group_keys=False).apply(lambda g: g.sample(min(len(g), quota), random_state=SEED))

    pos_df = balanced(eligible[eligible.topic == topic], "emotion", args.n)
    neg_df = balanced(eligible[eligible.topic != topic], "topic", args.n)
    rng = np.random.default_rng(SEED)

    def train_test(df):
        idx = rng.permutation(len(df))
        a, b = int(0.6 * len(df)), int(0.8 * len(df))
        return df.iloc[idx[:a]], df.iloc[idx[b:]]

    pos_tr, pos_te = train_test(pos_df)
    neg_tr, neg_te = train_test(neg_df)
    model, processor = load_model(args.model)
    acts = {k: story_activations(model, processor, d.story.tolist())
            for k, d in dict(pos_tr=pos_tr, pos_te=pos_te, neg_tr=neg_tr, neg_te=neg_te).items()}
    directions = build_direction(acts["pos_tr"], acts["neg_tr"], A["neu_tr"], args.pca)
    name = "topic"
    info.update(kind="topic", topic=topic, picked_automatically=args.topic == "auto",
                n_train=[len(pos_tr), len(neg_tr)], n_test=[len(pos_te), len(neg_te)],
                auc_topic_vs_other_topics=auc(acts["pos_te"][:, layer], acts["neg_te"][:, layer], directions[layer]))

d = directions[layer]
info.update(name=name, layer=layer,
            auc_sad_vs_other=auc(A["sad_te"][:, layer], A["oth_te"][:, layer], d),
            cosine_with_sad=float(d @ sad_dir[layer] / np.linalg.norm(sad_dir[layer])))
torch.save({"directions": torch.from_numpy(directions), "emotion": name}, f"{args.out_dir}/directions.pt")
json.dump(info, open(f"{args.out_dir}/control_info.json", "w"), indent=2)
print(json.dumps(info, indent=2))
