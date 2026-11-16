"""Target Planner（MBRL 层，慢环）：在世界模型内 rollout 搜目标轨迹。
输出 x_target ∈ ODD，代价 J = J_perf + β·J_unc + γ·J_risk。"""
import numpy as np

class TargetPlanner:
    def __init__(self, world, odd=None, beta=0.5, gamma=2.0, n_cand=200, seed=0):
        self.world = world; self.beta = beta; self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.odd = odd or dict(core_Te=(500.0, 5000.0))
        self.n_cand = n_cand
    def plan(self, X, u_span, mission_te, H=5):
        """候选目标网格 → 世界模型 rollout H 步 → 选 J 最小的目标核心温度。"""
        lo, hi = self.odd["core_Te"]
        cands = np.linspace(max(lo, mission_te * 0.6), min(hi, mission_te * 1.4), self.n_cand)
        best, bestJ = None, np.inf
        for xt in cands:
            Xr = X.copy(); J_unc = 0.0
            for k in range(H):                       # rollout 到目标评估代价
                u = np.clip((xt - Xr[0]) * u_span, -1, 1)
                _, Xr = self.world.step(Xr, u)
                J_unc += float(np.std(Xr))
            J_perf = abs(Xr[0] - xt) + 0.1 * abs(xt - mission_te)
            J_risk = max(0.0, Xr[0] - hi) ** 2       # 越 ODD 上界的风险
            J = J_perf + self.beta * J_unc + self.gamma * J_risk
            if J < bestJ:
                best, bestJ = xt, J
        return float(best), float(bestJ)

if __name__ == "__main__":
    from ..neural.world import DualWorldModel
    n = 30
    wm = DualWorldModel(n)
    wm._x_mu, wm._x_sd = np.zeros(n + 2), np.ones(n + 2)     # 无残差时单位归一
    wm._y_mu, wm._y_sd = np.zeros(n), np.ones(n)
    X0 = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    xt, J = TargetPlanner(wm).plan(X0, u_span=0.02, mission_te=2500.0)
    assert 1500 < xt < 3500
    print(f"target OK: x_target={xt:.0f} eV, J={J:.2f}")
