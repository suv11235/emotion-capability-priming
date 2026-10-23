import argparse
import collections
import json
import math

from common import mcnemar_p, outcome

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

ap = argparse.ArgumentParser()
ap.add_argument("records", nargs="+")
ap.add_argument("--baseline", default="sad")
ap.add_argument("--out", default="compare.md")
ap.add_argument("--fig", default=None)
a = ap.parse_args()

cond = collections.OrderedDict()
for path in a.records:
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["direction"], float(r["alpha_multiplier"]))
        q = str(r["question_id"])
        if q in cond.get(key, {}):
            continue
        o = outcome(r["response"], r["gold"], r["truncated"])
        o["tokens"] = r["n_new_tokens"]
        cond.setdefault(key, {})[q] = o

base = cond[(a.baseline, 0.0)]
rows = []
for (d, m), recs in sorted(cond.items(), key=lambda kv: (kv[0][0] != a.baseline, kv[0][0], kv[0][1])):
    n = len(recs)
    k = sum(o["correct"] for o in recs.values())
    shared = [q for q in recs if q in base]
    rw = sum(base[q]["correct"] and not recs[q]["correct"] for q in shared)
    wr = sum(not base[q]["correct"] and recs[q]["correct"] for q in shared)
    rows.append(dict(d=d, m=m, n=n, acc=k / n, ci=wilson(k, n),
                     diff=k / n - sum(base[q]["correct"] for q in shared) / len(shared),
                     rw=rw, wr=wr, p=mcnemar_p(rw, wr),
                     parse=sum(o["parse_failure"] for o in recs.values()) / n,
                     trunc=sum(o["truncated"] for o in recs.values()) / n,
                     tokens=sum(o["tokens"] for o in recs.values()) / n))

lines = ["95% CI = Wilson interval of each accuracy (not of the change). McNemar p uncorrected for multiple comparisons.", "",
         "| direction | mult | n | accuracy | 95% CI | change | right>wrong | wrong>right | McNemar p | parse fail | truncated | mean tokens |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['d']} | {r['m']:+g} | {r['n']} | {r['acc']:.3f} | {r['ci'][0]:.3f} to {r['ci'][1]:.3f} | "
                 f"{r['diff']:+.3f} | {r['rw']} | {r['wr']} | {r['p']:.3g} | {r['parse']:.3f} | {r['trunc']:.3f} | {r['tokens']:.0f} |")
open(a.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

if a.fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = collections.defaultdict(list)
    for r in rows:
        by[r["d"]].append(r)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for d, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["m"])
        x = [r["m"] for r in rs]
        for i, key in enumerate(["acc", "trunc", "tokens"]):
            y = [100 * r[key] if key != "tokens" else r[key] for r in rs]
            style = dict(fmt="-o" if len(rs) > 1 else "o", lw=2.5 if d == a.baseline else 1.2, label=d)
            if key == "acc":
                err = [[100 * (r["acc"] - r["ci"][0]) for r in rs], [100 * (r["ci"][1] - r["acc"]) for r in rs]]
                ax[i].errorbar(x, y, yerr=err, capsize=3, **style)
            else:
                ax[i].errorbar(x, y, **style)
    b = next(r for r in rows if r["d"] == a.baseline and r["m"] == 0.0)
    ax[0].axhspan(100 * b["ci"][0], 100 * b["ci"][1], color="grey", alpha=0.15, label="no-push 95% CI")
    for axis, title in zip(ax, ["Accuracy (%)", "Hit token limit (%)", "Mean tokens"]):
        axis.set_xlabel("push strength (multiples of delta)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=150)
    print(f"wrote {a.fig}")
