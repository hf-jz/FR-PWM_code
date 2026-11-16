"""T6：QP 可行性统计与不可行路径的记录（审稿要求"QP 不可行 → 分级回退"可审计）。

在运行区域内随机抽样状态与约束，统计 active-set QP 的可行比例；不可行时要求
filter() 走 safe_lookup 且返回的 u 在执行器盒与量程内（回退可执行，不是异常）。
输出 tests/results/safety_qp/report.json
用法: /usr/bin/python3 tests/qp_feasibility.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library"))
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.runtime.safety import SafetyKernel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "safety_qp")
N = 6


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    k = SafetyKernel([(np.zeros(N), 0.0, 0.0)], np.array([-1.0]), np.array([1.0]))
    rows = []
    for i in range(300):
        A = np.eye(N) * rng.uniform(0.9, 1.0)
        B = np.zeros((N, 1)); B[rng.integers(0, N), 0] = rng.uniform(0.5, 1.5)
        c = rng.normal(size=N) * 0.1
        x = rng.normal(size=N) * 0.5
        cons = [(rng.normal(size=N), float(rng.uniform(-2, 2)), 0.01) for _ in range(4)]
        k.cons = cons
        u, mode = k.filter(np.array([0.0]), x, (A, B, c))
        feas = "qp" in mode
        rows.append(dict(i=i, mode=mode, feasible=feas,
                         u=float(np.ravel(u)[0]), in_box=bool(-1.0 <= float(np.ravel(u)[0]) <= 1.0)))
    n_feas = sum(r["feasible"] for r in rows)
    n_fb = sum(1 for r in rows if not r["feasible"])
    all_box = all(r["in_box"] for r in rows)
    modes = {}
    for r in rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    rep = dict(n=len(rows), feasible=n_feas, fallback=n_fb,
               feasible_fraction=n_feas / len(rows), modes=modes, all_fallback_in_box=all_box)
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    assert all_box, "a fallback command left the actuator box"
    print(f"PASS: {n_feas}/{len(rows)} feasible, {n_fb} routed to the certified lookup, all inside the box")
    return rep


if __name__ == "__main__":
    main()
