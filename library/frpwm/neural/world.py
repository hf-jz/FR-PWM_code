"""双空间世界模型（报告 11.6/16：残差落物理态 + JEPA 隐空间交叉校验）。
X̂^P = F_P(X,U,M)，R_φ 物理空间残差，X̂^mix = F_P + R_φ；
ẑ = E(X̂^mix) vs z^JEPA = P_ψ(z,U,M) → Δz、ΔX 交叉校验；E_H^max < ε_H 验收。"""
import numpy as np
from ..nn import MLP

class PhysicsStep:
    """F_P：物理前向——扩散平滑 + 加热响应 + 衰减（1D 剖面，量纲一致）。"""
    def __init__(self, n, diff=0.02, decay=0.01, heat_gain=200.0):
        self.n, self.diff, self.decay, self.heat_gain = n, diff, decay, heat_gain
    def __call__(self, X, u=0.0):
        lap = np.roll(X, 1) - 2 * X + np.roll(X, -1)
        lap[0] = 2 * (X[1] - X[0]); lap[-1] = 2 * (X[-2] - X[-1])
        return X + self.diff * lap - self.decay * (X - X.min()) + self.heat_gain * u * self._dep()
    def _dep(self):
        r = np.linspace(0, 1, self.n)
        return np.exp(-((r - 0.3) / 0.25) ** 2)          # 加热沉积（芯部偏外）

class DualWorldModel:
    def __init__(self, n, d_z=32, jepa=None, seed=0):
        self.FP = PhysicsStep(n)
        self.R = MLP(n + 2, [64, 64], n, seed=seed)      # R_φ(X,z_mean,U) → 物理空间残差（eV/s 量纲）
        self.jepa = jepa                                  # PlasmaJepa（隐空间预测）
        self.n = n
    def train_residual(self, traj, epochs=200):
        """traj: [(X_t, u_t, X_{t+1})]，学 R_φ ≈ X_{t+1} − F_P(X_t,u)（物理未建模部分）。
        输入/目标归一化（X~10³、残差~10¹ 量级差，不归一 tanh 学不动）。"""
        X0 = np.array([t[0] for t in traj]); U = np.array([[t[1], 0.0] for t in traj])
        Y = np.array([t[2] for t in traj]) - np.array([self.FP(t[0], t[1]) for t in traj])
        Xf = np.concatenate([X0, U], -1)
        self._x_mu, self._x_sd = Xf.mean(0), Xf.std(0) + 1e-9
        self._y_mu, self._y_sd = Y.mean(0), Y.std(0) + 1e-9
        Xn, Yn = (Xf - self._x_mu) / self._x_sd, (Y - self._y_mu) / self._y_sd
        hist = [self.R.train_step(Xn, Yn) for _ in range(epochs)]
        return hist
    def clip_residual(self, r, ood=0.0):
        """神经残差可信域（2.1.3 冻结）：‖R_φ‖_{Σ⁻¹} ≤ B_R(Q,OOD,M)，OOD↑⇒B_R↓（满档→0）。
        实现 = 对 R_φ 输出投影/裁剪（residual clipping），越界部分截断。返回 (r_clip, clipped)。"""
        from .. import constants
        b_r = float(constants.get("b_r", 3.0)) * max(0.0, 1.0 - float(ood))
        r = np.asarray(r, float)
        sd = np.abs(self._y_sd) + 1e-30                 # Σ⁻¹ 范数用训练残差尺度（对角压缩）
        norm = float(np.sqrt(((r / sd) ** 2).sum()))
        if b_r <= 0.0:
            return np.zeros_like(r), norm > 0.0
        if norm > b_r:
            return r * (b_r / norm), True
        return r, False
    def step(self, X, u=0.0, ood=0.0):
        xP = self.FP(X, u)
        xn = (np.concatenate([X, [u, 0.0]]) - self._x_mu) / self._x_sd
        r = self.R.forward(xn)[0] * self._y_sd + self._y_mu
        r, _ = self.clip_residual(r, ood=ood)           # 可信域裁剪（Physics Grounded）
        xMix = xP + r
        return xP, xMix
    def cross_check(self, X, xMix, u=0.0):
        """Δz = ‖E(X̂^mix) − P_ψ(z)‖，ΔX = ‖X̂^mix − X̂^P‖（无 JEPA 时 Δz=None）。
        JEPA 观测=Y 抽稀网格（训练同形），不是全 120 维剖面——维度必须对齐。"""
        xP = self.FP(X, u)
        dX = float(np.linalg.norm(xMix - xP))
        dz = None
        if self.jepa is not None and self.jepa.dec is not None:
            d_y = self.jepa.enc.W[0].shape[0]           # JEPA 训练观测维
            st = max(1, len(X) // d_y)
            z = self.jepa.encode(np.atleast_2d(X[::st][:d_y]))
            z_jepa = self.jepa.predict(z, u_eff=u)
            z_mix = self.jepa.encode(np.atleast_2d(xMix[::st][:d_y]))
            dz = float(np.linalg.norm(z_mix - z_jepa))
        return dz, dX
    def h_step_error(self, X0, u, X_true_H, H):
        """E_H = (1/H)Σ‖X_k−X̂_k‖，E_H^max，对照 ε_H。"""
        X = X0.copy(); errs = []
        for k in range(H):
            _, X = self.step(X, u)
            errs.append(float(np.linalg.norm(X - X_true_H[k]) / (np.linalg.norm(X_true_H[k]) + 1e-30)))
        return float(np.mean(errs)), float(np.max(errs))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    # 合成动力学：物理项 + 未建模项（芯部额外源）
    FP = PhysicsStep(n)
    def true_step(X, u):
        extra = 30.0 * np.exp(-((np.linspace(0, 1, n) - 0.15) / 0.1) ** 2) * u
        return FP(X, u) + extra
    X0 = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    traj = []
    X = X0.copy()
    for _ in range(400):
        u = rng.uniform(-1, 1)
        Xn = true_step(X, u)
        traj.append((X.copy(), u, Xn.copy())); X = Xn
    wm = DualWorldModel(n)
    h = wm.train_residual(traj, epochs=100)
    assert h[-1] < 0.2 * h[0]
    xP, xMix = wm.step(X0, 0.8)
    xT = true_step(X0, 0.8)
    err_mix = np.linalg.norm(xMix - xT) / np.linalg.norm(xT)
    err_P = np.linalg.norm(xP - xT) / np.linalg.norm(xT)
    assert err_mix < err_P and err_mix < 0.05, (err_P, err_mix)   # 残差修正有效
    Xs = [X0.copy()]
    X = X0.copy()
    for _ in range(10):
        X = true_step(X, 0.5); Xs.append(X.copy())
    eH, eHmax = wm.h_step_error(X0, 0.5, Xs[1:], 10)
    assert eHmax < 0.10, eHmax                                      # ε_H(H=10)=10% 预算内
    # 2.1.3 可信域：OOD 满档 → B_R→0 → xMix≡xP（纯物理主导）
    xP1, xM1 = wm.step(X0, 0.8, ood=1.0)
    assert np.allclose(xP1, xM1)
    # 巨大残差必被裁剪到 B_R 内
    r_big, clipped = wm.clip_residual(np.full(n, 1e6), ood=0.0)
    assert clipped and np.sqrt(((r_big / (np.abs(wm._y_sd) + 1e-30)) ** 2).sum()) <= 3.0 + 1e-9
    print(f"worldmodel OK: train {h[0]:.4f}->{h[-1]:.5f}, mix err {err_mix:.4f} < P err {err_P:.4f}, E_10^max={eHmax:.3f}, B_R clip OK")
