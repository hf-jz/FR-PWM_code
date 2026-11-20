"""Post-lock functional-check pass over the ten campaign discharges.

The 900-check tally in the draft comes from a run written before the criteria were locked, and the sixty-shot pool it
used is no longer on this machine (the tokamark-v1 archive ships as two-gigabyte tar batches). The ten campaign
discharges are served directly by FAIR-MAST, so the same fifteen checks per discharge are re-run here on the current
code, and the tally is reported with its own scope rather than merged with the sixty-shot record.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage12"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from fairmast_zarr import convert  # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
import demo_mast  # noqa: E402

POOL = "/tmp/mast28"
SHOTS = [str(s) for s in range(28315, 28325)]
OUT = "stage40/postfreeze_checks.json"
os.makedirs(POOL, exist_ok=True)

rows = []
for shot in SHOTS:
    zp = os.path.join(POOL, "%s.zarr" % shot)
    if not os.path.isdir(zp):
        try:
            convert(shot, outdir=POOL)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(shot=shot, error="fetch: %s" % str(exc)[:100]))
            print(shot, "fetch failed", str(exc)[:80], flush=True)
            continue
    try:
        P, probs, M = demo_mast.check_shot(zp, n_train=60, n_online=40)
    except Exception as exc:  # noqa: BLE001
        rows.append(dict(shot=shot, error="run: %s" % str(exc)[:100]))
        print(shot, "run failed", str(exc)[:80], flush=True)
        continue
    ok = sum(1 for _, v, _ in P if v)
    rows.append(dict(shot=shot, checks=len(P), passed=ok,
                     failed=[n for n, v, _ in P if not v], metrics=M))
    print("%s: %d/%d passed, approved %d/%d, P95 %.1f ms"
          % (shot, ok, len(P), M["approved"], M["approved"] + M["rejected"], M["p95_ms"]), flush=True)

total = sum(r.get("checks", 0) for r in rows)
passed = sum(r.get("passed", 0) for r in rows)
out = dict(scope="ten campaign discharges, post-lock code, fifteen checks per discharge",
           shots=len([r for r in rows if "checks" in r]), checks=total, passed=passed,
           failed_names=sorted({n for r in rows for n in r.get("failed", [])}),
           rows=rows)
os.makedirs("stage40", exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print("TOTAL %d/%d checks passed over %d discharges; wrote %s"
      % (passed, total, out["shots"], OUT))
