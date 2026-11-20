"""B3 诊断消融扫描：逐组摘除诊断，测物理量误差与报告不确定度的响应。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
import json, os, subprocess, sys, time

PY = "/usr/bin/python3"
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library")
OUT = os.path.join(HERE, "results", "ablation")
CONFIGS = ["", "euv", "bal", "brem", "ph", "euv,bal", "euv,bal,brem", "euv,bal,brem,ph"]
os.makedirs(OUT, exist_ok=True)

for cfg in CONFIGS:
    name = cfg.replace(",", "+") or "none"
    dst = os.path.join(OUT, f"{name}.json")
    if os.path.exists(dst):
        print("skip", name, flush=True)
        continue
    env = dict(os.environ, ABL_DROP=cfg, NFOLDS="3", NSEEDS="2", ABL_OUT=dst)
    t0 = time.time()
    r = subprocess.run([PY, "diag_ablate.py"], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=1200)
    print(f"{name:22s} {time.time()-t0:6.0f}s rc={r.returncode} "
          f"{'OK' if os.path.exists(dst) else 'FAIL'}", flush=True)
    if not os.path.exists(dst):
        print(r.stdout[-500:], r.stderr[-500:], flush=True)
print("ALL DONE")
