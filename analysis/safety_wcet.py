"""Safety-kernel timing artifact: P50/P99/WCET of the CBF-QP kernel (C++ backend).

Writes results/safety_wcet.json so the number quoted for the shield has a stored source.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frpwm.runtime.safety_cpp import cbf_qp, _HAS_CPP

rng = np.random.default_rng(0)
n, m = 3, 4
G = rng.normal(0, 1, (m, n)); h = np.abs(rng.normal(0, 1, m)) + 0.5
uc = rng.normal(0, 2, n); lo, hi = -1.5 * np.ones(n), 1.5 * np.ones(n)
for _ in range(50):
    cbf_qp(uc, G, h, lo, hi)                      # warm-up
ts = []
for _ in range(2000):
    t0 = time.perf_counter()
    cbf_qp(uc, G, h, lo, hi)
    ts.append((time.perf_counter() - t0) * 1e6)
a = np.sort(np.array(ts))
out = dict(backend="cpp" if _HAS_CPP else "scipy",
           n=len(a), P50=float(np.percentile(a, 50)), P99=float(np.percentile(a, 99)),
           WCET=float(a.max()), problem=dict(n_state=n, n_constraint=m))
json.dump(out, open("../tests/results/safety_wcet.json", "w"), indent=1)
print(json.dumps(out, indent=1))
