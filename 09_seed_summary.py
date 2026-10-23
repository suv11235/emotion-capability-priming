import argparse
import collections
import json
import numpy as np

from common import outcome, parse_answer

ap = argparse.ArgumentParser()
ap.add_argument("records", nargs="+", help="one records.jsonl per seed")
ap.add_argument("--out", default="seeds.md")
a = ap.parse_args()

runs = []
for path in a.records:
    cond, seeds = collections.defaultdict(dict), set()
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            seeds.add(r.get("sample_seed"))
            cond[(r["direction"], float(r["alpha_multiplier"]))][str(r["question_id"])] = (
                outcome(r["response"], r["gold"], r["truncated"])["correct"], parse_answer(r["response"]), r["gold"])
    if None in seeds or len(seeds) != 1:
        raise SystemExit(f"{path}: expected one sampled seed, found {seeds}")
    runs.append(cond)

keys = sorted(set.intersection(*[set(c) for c in runs]))
base_key = next(k for k in keys if k[1] == 0.0)

def per_question(key):
    qs = sorted(set.intersection(*[set(c[key]) for c in runs]))
    return qs, np.array([[c[key][q][0] for q in qs] for c in runs], float)

bq, bmat = per_question(base_key)
bmean = dict(zip(bq, bmat.mean(0)))
rng = np.random.default_rng(0)
lines = ["| direction | mult | seeds | n | mean acc | seed SD | change | 95% CI | pass@k | majority | per-seed |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for key in keys:
    qs, mat = per_question(key)
    acc = mat.mean(1)
    majority = []
    for q in qs:
        votes = collections.Counter(c[key][q][1] for c in runs if c[key][q][1] is not None).most_common()
        win = votes[0][0] if votes and (len(votes) == 1 or votes[0][1] > votes[1][1]) else None
        majority.append(win is not None and win == runs[0][key][q][2])
    change = ci = ""
    if key != base_key:
        shared = [i for i, q in enumerate(qs) if q in bmean]
        s, b = mat.mean(0)[shared], np.array([bmean[qs[i]] for i in shared])
        idx = rng.integers(0, len(s), size=(2000, len(s)))
        diffs = s[idx].mean(1) - b[idx].mean(1)
        change = f"{s.mean() - b.mean():+.3f}"
        ci = f"{np.quantile(diffs, 0.025):+.3f} to {np.quantile(diffs, 0.975):+.3f}"
    lines.append(f"| {key[0]} | {key[1]:+g} | {len(runs)} | {len(qs)} | {acc.mean():.3f} | "
                 f"{acc.std(ddof=1) if len(runs) > 1 else 0:.3f} | {change} | {ci} | {mat.max(0).mean():.3f} | "
                 f"{np.mean(majority):.3f} | {', '.join(f'{x:.3f}' for x in acc)} |")
open(a.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
