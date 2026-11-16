"""C++ Safety Kernel 桥接（frpwm_cpp 优先，不可用时纯 Python 回退——同一数学语义）。
离散时间 CBF-QP：min ½‖u−u_cand‖² s.t. G u ≤ h, lo≤u≤hi；失败 → Safe Lookup（O(1）快速路径）。"""
import numpy as np

try:
    from .. import frpwm_cpp
    _HAS_CPP = hasattr(frpwm_cpp, "cbf_qp")
except Exception:  # noqa: BLE001
    frpwm_cpp, _HAS_CPP = None, False

def cpp_available():
    return _HAS_CPP

def cbf_qp(u_cand, G, h, lo, hi, maxiter=0):
    """→ (u, ok, backend)。C++ active-set 优先；不可用/失败 → scipy SLSQP 回退。"""
    u_cand = np.asarray(u_cand, float)
    G, h = np.asarray(G, float), np.asarray(h, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if _HAS_CPP:
        u, ok = frpwm_cpp.cbf_qp(u_cand, G, h, lo, hi, maxiter)
        if ok and np.all(G @ np.asarray(u) <= h + 1e-7):
            return np.asarray(u), True, "cpp"
    from scipy.optimize import minimize
    u0 = np.clip(u_cand, lo, hi)
    try:
        r = minimize(lambda u: 0.5 * float(((u - u0) ** 2).sum()), u0,
                     jac=lambda u: u - u0, bounds=list(zip(lo, hi)),
                     constraints=[{"type": "ineq", "fun": lambda u: h - G @ u}],
                     method="SLSQP", options={"maxiter": 50, "ftol": 1e-12})
        if r.success and np.all(G @ r.x <= h + 1e-9):
            return r.x, True, "scipy"
    except Exception:  # noqa: BLE001
        pass
    return u0, False, "fallback"

if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    # 与随机 QP 对拍：C++ 与 scipy 解一致（1e-6）
    for trial in range(30):
        n, m = 3, 4
        G = rng.normal(0, 1, (m, n))
        h = np.abs(rng.normal(0, 1, m)) + 0.5
        uc = rng.normal(0, 2, n)
        lo, hi = -np.ones(n) * 1.5, np.ones(n) * 1.5
        u1, ok1, be1 = cbf_qp(uc, G, h, lo, hi)
        assert ok1
        from scipy.optimize import minimize
        r = minimize(lambda u: 0.5 * ((u - np.clip(uc, lo, hi)) ** 2).sum(), np.clip(uc, lo, hi),
                     jac=lambda u: u - np.clip(uc, lo, hi), bounds=list(zip(lo, hi)),
                     constraints=[{"type": "ineq", "fun": lambda u: h - G @ u}],
                     method="SLSQP", options={"ftol": 1e-12})
        assert np.allclose(u1, r.x, atol=1e-5), (trial, u1, r.x, be1)
    # 不可行案例 → ok=False
    G2 = np.array([[1.0]]); h2 = np.array([-2.0])
    u2, ok2, _ = cbf_qp(np.array([0.0]), G2, h2, np.array([-1.0]), np.array([1.0]))
    assert not ok2
    # 时延（P99 < 1ms 预算，实测应 ≪0.1ms）
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        cbf_qp(uc, G, h, lo, hi)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    print(f"safety_cpp OK: backend={'cpp' if _HAS_CPP else 'scipy'} 对拍 30/30, "
          f"P99={ts[int(0.99*len(ts))]:.1f}µs")
