# -*- coding: utf-8 -*-
"""Shared Spectral Encoder/Species Heads 段 WCET 实测（plan_eval09121836 ⑦收口）。
原 JEPA/Head=1.0/1.5ms 设计预算 → 新网络（PGMZeff：ConvStem+SpectralTransformer+
PhysicsToken 融合+全部物种头）CPU 实测 P99/WCET，产出 tests/results/wcet_upsse/report.json。
run_all 子进程运行；断言 P99<1.5ms 设计预算（超预算则 FAIL 并须降级/优化后再冻结）。"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library"))
import torch  # noqa: E402

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.neural.pgm_zeff import PGMZeff  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() // 2))

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artefacts", "A9_A10_Timing")
z = np.load(os.path.join(BASE, "train_w96.npz"))
X = torch.tensor(np.log1p(z["frames"][:8]).astype(np.float32))
N = X.shape[0]
phys = torch.tensor(np.tile(np.array([13.9 / 14, 3.63 / 4, 0.5, 1.0, 0.55], np.float32), (N, 1)))
brem = torch.ones(N, 1)

model = PGMZeff(n_q=75)
model.eval()

with torch.no_grad():
    for _ in range(20):                                  # warmup（含 cudnn/allocator 冷启动）
        model(X, phys, brem)
    ts = []
    for _ in range(300):
        t0 = time.perf_counter()
        model(X, phys, brem)
        ts.append((time.perf_counter() - t0) * 1000 / N)  # 摊薄到单帧
    # 单帧串行实测（贴近在线语义，非批摊薄）
    ts1 = []
    for _ in range(300):
        i = np.random.randint(N)
        t0 = time.perf_counter()
        model(X[i:i + 1], phys[i:i + 1], brem[i:i + 1])
        ts1.append((time.perf_counter() - t0) * 1000)

def stats(a):
    a = np.asarray(a)
    return dict(P50=float(np.percentile(a, 50)), P99=float(np.percentile(a, 99)),
                WCET=float(a.max()))

res = dict(batch8_per_frame=stats(ts), single_frame=stats(ts1))
print(json.dumps(res, indent=1))
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "wcet_upsse_report.json"), "w"), indent=1)

p99 = res["single_frame"]["P99"]
print(f"Shared Spectral Encoder/Species Heads 单帧 P99 = {p99:.3f} ms（设计预算 1.5 ms）")
ok = p99 < 1.5
print("PASS" if ok else "FAIL：超设计预算，须优化/降级后重测")
sys.exit(0 if ok else 1)
