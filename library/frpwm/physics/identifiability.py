"""可辨识性分级 + 五源不确定度（P0-4，报告 17 治理：Fisher/条件数 → I_t^ident；Σ 五源分解）。"""
import numpy as np

def fisher_grade(forward, theta, dy_scale=None, h=1e-4, kappa=(1e3, 1e6)):
    """Fisher 信息（数值 Jacobian J†J）条件数 → I_t^ident ∈ {0 可辨识,1 弱,2 不可辨识}。
    返回 (grade, cond, singular_values)。"""
    theta = np.atleast_1d(np.asarray(theta, float))
    f0 = np.atleast_1d(forward(theta))
    J = np.zeros((len(f0), len(theta)))
    for j in range(len(theta)):
        d = np.zeros_like(theta); d[j] = h * max(abs(theta[j]), 1e-6)
        J[:, j] = (np.atleast_1d(forward(theta + d)) - f0) / d[j]
    if dy_scale is not None:
        J = J / np.asarray(dy_scale)[:, None]
    F = J.T @ J
    s = np.linalg.svd(F, compute_uv=False)
    cond = float(np.sqrt(s.max() / max(s.min(), 1e-300)))
    grade = 0 if cond < kappa[0] else (1 if cond < kappa[1] else 2)
    return grade, cond, s

def sigma_five(meas=0.0, cal=0.02, epi=0.0, mf=0.20, proc=0.0):
    """五源分解：测量/标定/认知/模型形式(原子库~20%)/过程。返回 dict + total。"""
    s = dict(meas=float(meas), cal=float(cal), epi=float(epi), mf=float(mf), proc=float(proc))
    s["total"] = float(np.sqrt(sum(v ** 2 for k, v in s.items() if k != "total")))
    return s

if __name__ == "__main__":
    # 可辨识：Y=θ0+θ1（两参数各向异性）
    f1 = lambda th: np.array([th[0] + th[1], th[0] - th[1], th[0] * th[1]])
    g1, c1, _ = fisher_grade(f1, np.array([0.5, -0.8]))
    # 简并：Y=θ0·θ1（折叠简并方向）
    f2 = lambda th: np.array([th[0] * th[1], (th[0] * th[1]) ** 2])
    g2, c2, _ = fisher_grade(f2, np.array([1.0, 1.0]))
    assert g1 == 0 and g2 >= 1, (c1, c2)
    s = sigma_five(meas=0.05, epi=0.1, proc=0.03)
    assert abs(s["total"] - np.sqrt(0.05**2 + 0.02**2 + 0.1**2 + 0.2**2 + 0.03**2)) < 1e-9
    print(f"identifiability OK: 可辨识 cond={c1:.1f}(g{g1}) 简并 cond={c2:.1e}(g{g2}) total_σ={s['total']:.3f}")
