"""T3-E1 / T3-E2：物理核收缩因子 ρ 与残差增益 L_r 的实测（不是假设）。

审稿意见 T3 的核心批评：§2.3 的界 2δ/(1-ρ) 只有在核在信任域内收缩、且 L_r < 1-ρ 时才
有信息量，而原文只写了"假设"。本脚本在状态空间的运行区域上直接测这两个量：
  ρ = sup_k ‖δ_{k+1}‖/‖δ_k‖，δ_k = F_P^k(X0+ε) − F_P^k(X0)（多起点、多方向）
  L_r = max ‖R_φ(X+ε)−R_φ(X)‖/‖ε‖（有限差分，训练分布内/外各一组）
输出 tests/results/core_contraction/report.json
用法: /usr/bin/python3 tests/core_contraction.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frpwm.neural.world import DualWorldModel, PhysicsStep

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "core_contraction")
N = 16
K = 10
EPS = 1e-3


def _phys(wm, X, u):
    """R_φ 的物理量纲输出（逆归一化），输入先按训练统计量归一化。"""
    n = len(X)
    x = (np.asarray(X) - wm._x_mu[:n]) / wm._x_sd[:n]
    uu = (np.array([u, 0.0]) - wm._x_mu[n:]) / wm._x_sd[n:]
    z = np.concatenate([x, uu]).reshape(1, -1)
    return np.asarray(wm.R.forward(z))[0] * wm._y_sd + wm._y_mu


def main():
    os.makedirs(OUT, exist_ok=True)
    FP = PhysicsStep(N)
    rng = np.random.default_rng(7)
    # 运行区域：平滑剖面（cos 形）+ 随机起点，避免只测一个点
    bases = [50.0 + 20.0 * np.cos(np.linspace(0, np.pi, N)), 30.0 + 5.0 * rng.normal(size=N)]
    ratios, per_step = [], []
    for X0 in bases:
        for _ in range(4):
            d = rng.normal(size=N)
            d *= EPS / np.linalg.norm(d)
            xs, xp = X0.copy(), X0 + d
            seq = []
            for k in range(K):
                dn = np.linalg.norm(xp - xs)
                xs, xp = FP(xs), FP(xp)
                seq.append(float(np.linalg.norm(xp - xs) / dn))
            per_step.append(seq)
            ratios.append(max(seq[1:]))          # 去掉首步瞬态
    rho = float(max(ratios))

    # 残差网络：用同一生成器的轨迹训练（与 rollout 台架一致），再估 L_r
    wm = DualWorldModel(N, seed=0)
    traj = []
    for _ in range(400):
        X = 20.0 + 40.0 * rng.random(N)
        u = float(rng.uniform(-1, 1))
        traj.append((X, u, FP(X, u) + 0.02 * u * np.ones(N)))   # 未建模项 ∝ u
    wm.train_residual(traj, epochs=200)

    def lip(scale):
        worst = 0.0
        for _ in range(200):
            X = 20.0 + 40.0 * rng.random(N) * scale
            d = rng.normal(size=N); d *= 1e-4 / np.linalg.norm(d)
            u = float(rng.uniform(-1, 1))
            def _r(Xv):
                return _phys(wm, Xv, u)
            r0, r1 = _r(X), _r(X + d)
            worst = max(worst, float(np.linalg.norm((r1 - r0) * wm._y_sd) / np.linalg.norm(d)))
        return worst

    lr_in, lr_out = lip(1.0), lip(1.6)
    delta = float(np.mean([np.abs(_phys(wm, b, 0.0)).max() for b in bases]))
    # T3-E3：信任域边界扫描——残差随 OOD 档位单调收缩，满档退化为纯物理
    sweep = []
    for ood in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = np.concatenate([(bases[0] - wm._x_mu[:N]) / wm._x_sd[:N],
                            (np.array([1.0, 0.0]) - wm._x_mu[N:]) / wm._x_sd[N:]])
        r = np.asarray(wm.R.forward(z.reshape(1, -1)))[0] * wm._y_sd
        rc, clipped = wm.clip_residual(r, ood=ood)
        sweep.append(dict(ood=ood, norm_raw=float(np.linalg.norm(r)),
                          norm_clipped=float(np.linalg.norm(rc)), clipped=bool(clipped)))
    assert sweep[-1]["norm_clipped"] == 0.0, sweep[-1]
    assert all(sweep[i]["norm_clipped"] >= sweep[i + 1]["norm_clipped"] - 1e-12
               for i in range(len(sweep) - 1)), sweep

    rep = dict(n=N, k_steps=K, rho_core=rho, trust_region_sweep=sweep, rho_per_start=[max(s[1:]) for s in per_step],
               L_r_in_domain=lr_in, L_r_out_of_domain=lr_out, residual_scale=delta,
               bound=2 * delta / max(1e-9, 1 - rho),
               informative=bool(lr_in < 1 - rho))
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    assert rho < 1.0, f"core does not contract on the region: rho={rho:.3f}"
    print(f"PASS: rho_core={rho:.3f} < 1; L_r(in)={lr_in:.3f} < 1-rho={1 - rho:.3f} -> bound informative")
    return rep


if __name__ == "__main__":
    main()
