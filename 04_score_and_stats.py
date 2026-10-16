"""Step 4: score the raw answers and run the statistics. Runs on saved response records.

    python 04_score_and_stats.py out/mmlu/records.jsonl

Reads : records.jsonl from step 3 (or results/mmlu_sad/mmlu_records.jsonl)
Prints: a table  strength | accuracy | change | McNemar p | BH q | parse% | trunc%
Writes: <same folder>/scores.csv

Tests (all planned before the runs):
  * paired: every strength answers the same questions as strength 0
  * McNemar: only looks at questions whose right/wrong status changed
  * Benjamini-Hochberg: corrects across all nonzero strengths included in the file
  * quality gate: parse failures and truncations must each be under 10%
"""
import json
import sys
from math import comb
import numpy as np
import pandas as pd

from common import outcome

path = sys.argv[1]
direction = sys.argv[2] if len(sys.argv) > 2 else None
recs = [json.loads(line) for line in open(path)]
df = pd.DataFrame(recs)
if direction:
    df = df[df.direction == direction]
for k, v in pd.DataFrame([outcome(r.response, r.gold, r.truncated) for r in df.itertuples()]).items():
    if k != "truncated":
        df[k] = v.values


def mcnemar(base, cond):
    b = int((base & ~cond).sum())     # right at 0, wrong now
    c = int((~base & cond).sum())     # wrong at 0, right now
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(q, 1)
    return out


wide = df.pivot(index="question_id", columns="alpha_multiplier", values="correct").astype(bool)
base = wide[0.0].values
mults = [m for m in wide.columns if m != 0.0]
pvals = [mcnemar(base, wide[m].values) for m in mults]
qvals = dict(zip(mults, bh(pvals)))
pvals = dict(zip(mults, pvals))

rows = []
for m, g in df.groupby("alpha_multiplier"):
    rows.append(dict(
        strength=m, n=len(g), accuracy=g.correct.mean(),
        change_pts=100 * (g.correct.mean() - base.mean()) if m != 0 else np.nan,
        mcnemar_p=pvals.get(m, np.nan), bh_q=qvals.get(m, np.nan),
        parse_fail_pct=100 * g.parse_failure.mean(), truncated_pct=100 * g.truncated.mean(),
        mean_tokens=g.n_new_tokens.mean(),
        passes_gate=bool(g.parse_failure.mean() < .10 and g.truncated.mean() < .10)))
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
res.to_csv(path.rsplit("/", 1)[0] + "/scores.csv", index=False)

# curved (quadratic) fit: accuracy ~ a + b*x + c*x^2. Negative c = hill shape.
x, y = res.strength.values, res.accuracy.values
c2, c1, c0 = np.polyfit(x, y, 2)
r2 = 1 - ((y - np.polyval([c2, c1, c0], x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"\nquadratic term {c2:.4f}, R^2 {r2:.2f}, peak at strength {-c1 / (2 * c2):.2f}")
