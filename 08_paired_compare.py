import argparse
import json
import random

from common import mcnemar_p, outcome

ap = argparse.ArgumentParser()
ap.add_argument("records", nargs="+")
ap.add_argument("--pairs", nargs="+", required=True, help="A:B direction names; the difference is A - B")
ap.add_argument("--multipliers", default="-8,8")
ap.add_argument("--out", default="paired.md")
a = ap.parse_args()

cond = {}
for path in a.records:
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            key, q = (r["direction"], float(r["alpha_multiplier"])), str(r["question_id"])
            if q not in cond.get(key, {}):
                cond.setdefault(key, {})[q] = outcome(r["response"], r["gold"], r["truncated"])["correct"]

def bootstrap(xa, xb, n_boot=10000, seed=0):
    rng, d = random.Random(seed), [int(x) - int(y) for x, y in zip(xa, xb)]
    stats = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]

lines = ["| A | B | mult | shared | acc A | acc B | A - B | 95% CI | A right, B wrong | A wrong, B right | McNemar p |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for pair in a.pairs:
    da, db = pair.split(":")
    for m in (float(x) for x in a.multipliers.split(",")):
        ca, cb = cond.get((da, m)), cond.get((db, m))
        if not ca or not cb:
            lines.append(f"| {da} | {db} | {m:+g} | missing | | | | | | | |")
            continue
        qs = sorted(set(ca) & set(cb))
        xa, xb = [ca[q] for q in qs], [cb[q] for q in qs]
        x = sum(p and not q for p, q in zip(xa, xb))
        y = sum(q and not p for p, q in zip(xa, xb))
        lo, hi = bootstrap(xa, xb)
        acc_a, acc_b = sum(xa) / len(qs), sum(xb) / len(qs)
        lines.append(f"| {da} | {db} | {m:+g} | {len(qs)} | {acc_a:.3f} | {acc_b:.3f} | {acc_a - acc_b:+.3f} | "
                     f"{lo:+.3f} to {hi:+.3f} | {x} | {y} | {mcnemar_p(x, y):.3g} |")
open(a.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
