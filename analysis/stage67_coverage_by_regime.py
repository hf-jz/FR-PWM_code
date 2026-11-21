"""Coverage per transport regime, on the same 256 out-of-fold cases.

The manuscript reports one pooled coverage number for the nominal ninety per cent interval. A pooled number hides the
question a safety-critical user asks, which is whether the interval still covers in the part of the domain where the
transport parameters are extreme. The 256 archived cases carry their true diffusion coefficient and convection speed,
so the cases can be split into tertiles on either axis and the coverage recomputed inside each tertile with the same
quantile, without refitting anything.

Writes stage67/coverage_by_regime.json.
"""
import csv
import json
import os

import numpy as np

os.makedirs("stage67", exist_ok=True)
pred = {r["discharge_id"]: r for r in csv.DictReader(open("stage5/data/conformal_predictions.csv"))}
cases = list(csv.DictReader(open("stage5/data/transport_cases.csv")))

joined = []
for i, c in enumerate(cases):
    key = "case%d" % i
    r = pred.get(key)
    if r is None:
        continue
    joined.append(dict(D=float(c["D_true"]), V=float(c["V_true"]) if c["V_true"] else 0.0,
                       y=float(r["y_true"]), lo=float(r["lower"]), hi=float(r["upper"])))


def tertiles(vals):
    q = np.quantile(vals, [1 / 3, 2 / 3])
    return [(0, q[0]), (q[0], q[1]), (q[1], np.inf)]


out = {"cases": len(joined), "pooled": float(np.mean([j["lo"] <= j["y"] <= j["hi"] for j in joined]))}
for axis in ("D", "V"):
    vals = np.array([j[axis] for j in joined])
    buckets = []
    for k, (a, b) in enumerate(tertiles(vals)):
        sel = [j for j in joined if a <= j[axis] < b]
        if not sel:
            continue
        cov = float(np.mean([j["lo"] <= j["y"] <= j["hi"] for j in sel]))
        width = float(np.mean([j["hi"] - j["lo"] for j in sel]))
        buckets.append(dict(tertile=k + 1, lo=float(a), hi=float(b) if np.isfinite(b) else None,
                            n=len(sel), coverage=cov, mean_width=width))
        print("%s tertile %d [%.3g, %.3g) n=%3d coverage %.3f width %.3f"
              % (axis, k + 1, a, b if np.isfinite(b) else float("inf"), len(sel), cov, width))
    out["by_" + axis] = buckets

worst = min(b["coverage"] for k in ("by_D", "by_V") for b in out[k])
out["worst_tertile_coverage"] = float(worst)
json.dump(out, open("stage67/coverage_by_regime.json", "w"), indent=1)
print("pooled %.3f, worst tertile %.3f" % (out["pooled"], worst))
