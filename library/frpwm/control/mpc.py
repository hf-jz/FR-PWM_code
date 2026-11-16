"""Fast MPC（报告 16 层级控制：短时域 H_MPC∈[5,20]，1-10ms）+ 执行器约束。
CEM 采样优化；u_min≤u≤u_max、|Δu|≤r_max（slew rate）；状态约束由 Safety Shield 裁决。
2.1.3 horizon 命名冻结：H_MPC=控制时域（本模块），H_WM=世界模型验收时域，禁裸 H。"""
import numpy as np
from .. import constants

class FastMpc:
    def __init__(self, world, H_MPC=8, n_cand=128, u_min=-1.0, u_max=1.0, slew=0.2, seed=0, H=None):
        if H is not None:                       # 兼容旧裸 H 参数（弃用，命名冻结 2.1.3）
            H_MPC = H
        lo_c, hi_c = constants.get("h_mpc", (5, 20))
        assert lo_c <= H_MPC <= hi_c, f"H_MPC={H_MPC} 超出冻结时域 [{lo_c},{hi_c}]"
        self.world, self.H, self.n_cand = world, H_MPC, n_cand
        self.u_min, self.u_max, self.slew = u_min, u_max, slew
        self.rng = np.random.default_rng(seed)
        self.u_prev = 0.0
    def act(self, X, x_target, risk=None):
        """CEM：候选动作序列 → rollout → 选代价最小首步（满足限幅+slew）。
        u_prev 先钳进 [u_min,u_max]，防执行器历史越界导致可行域倒置。
        risk=(P_event, TTE) 时防御性收缩动作窗口（barrier 语义）：风险越高、
        TTE 越短，可用动作幅度越小（P2-17，报告安全收敛条款）。"""
        P_ev, tte = risk if risk else (0.0, None)
        shrink = 1.0 - 0.8 * P_ev * (1.0 / (tte + 0.5) if tte else 0.0)
        up = float(np.clip(self.u_prev, self.u_min, self.u_max))
        lo = max(self.u_min, up - self.slew) * shrink
        hi = min(self.u_max, up + self.slew) * shrink
        U = self.rng.uniform(lo, hi, (self.n_cand, self.H))
        J = np.zeros(self.n_cand)
        for i in range(self.n_cand):
            Xr = X.copy(); u_p = up
            for k in range(self.H):
                uk = np.clip(U[i, k], u_p - self.slew, u_p + self.slew)   # 逐段 slew
                _, Xr = self.world.step(Xr, uk)
                J[i] += (Xr[0] - x_target) ** 2 + 1e-3 * uk ** 2
                u_p = uk
        u0 = float(np.clip(U[np.argmin(J), 0], lo, hi))
        self.u_prev = u0
        return u0

if __name__ == "__main__":
    from ..neural.world import DualWorldModel, PhysicsStep
    n = 30
    wm = DualWorldModel(n)
    wm._x_mu, wm._x_sd = np.zeros(n + 2), np.ones(n + 2)
    wm._y_mu, wm._y_sd = np.zeros(n), np.ones(n)
    mpc = FastMpc(wm, H=6, slew=0.3)
    X = 1000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    for _ in range(40):                       # 跟踪 x_target=2500 eV 芯温
        u = mpc.act(X, 2500.0)
        X = wm.FP(X, u)
    assert abs(X[0] - 2500) / 2500 < 0.15, X[0]
    du = [0.0]
    X2 = X.copy(); mpc.u_prev = 0.0
    u_seq = [mpc.act(X2, 2500.0) for _ in range(5)]
    assert all(abs(a - b) <= 0.3 + 1e-9 for a, b in zip(u_seq[1:], u_seq[:-1]))  # slew 约束
    print(f"mpc OK: core Te {1000:.0f}->{X[0]:.0f} eV (target 2500), u={u_seq[:3]}")
