"""TRM 物理精修单元（报告 11.5/16/17：Learned Preconditioned Iterative Solver）。
θ_{k+1}=θ_k+α_k·c_k·TRM_φ(θ_k,r_k,h_k)，r_k=W_y[Y_t−H(F_P(θ_k))]，
Trust Region ‖Δθ‖≤δ_k + Armijo 线搜索 + merit 单调 + 回滚。
训练：教师优化器（数值 Gauss-Newton）轨迹 → (θ,r)→(Δθ,c) 监督。"""
import numpy as np
from ..nn import MLP
from .. import constants

class TrmSolver:
    def __init__(self, d_theta, d_r, seed=0):
        self.net = MLP(d_theta + d_r + 1, [48, 48], d_theta + 1, seed=seed)  # (Δθ, c)
        self.delta = constants.get("trust_delta0", 0.1)
        self.eps = constants.get("eps_resid", 0.01)

    # ---------- 训练 ----------
    def train_teacher(self, forward, theta_star, Y_fn, n_traj=300, k_max=8, seed=0):
        """教师=数值 GN：从扰动初值收敛到 θ* 的轨迹监督 Δθ。"""
        rng = np.random.default_rng(seed)
        X, Y = [], []
        for _ in range(n_traj):
            th = theta_star * (1 + 0.5 * rng.standard_normal(len(theta_star)))
            for k in range(k_max):
                r = self._resid(forward, th, Y_fn)
                if np.linalg.norm(r) < 1e-12:
                    break
                dth = self._gn_step(forward, th, Y_fn, r)
                X.append(np.concatenate([th, r, [1.0]]))
                Y.append(np.concatenate([dth, [1.0]]))       # c=1（教师全置信）
                th = th + dth
        X, Y = np.array(X), np.array(Y)
        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-9
        self._y_mu, self._y_sd = Y.mean(0), Y.std(0) + 1e-9
        hist = [self.net.train_step((X - self._x_mu) / self._x_sd, (Y - self._y_mu) / self._y_sd)
                for _ in range(200)]
        return hist

    @staticmethod
    def _resid(forward, th, Y_fn):
        return Y_fn() - forward(th)

    @staticmethod
    def _gn_step(forward, th, Y_fn, r, h=1e-4):
        J = np.zeros((len(r), len(th)))
        f0 = forward(th)
        for j in range(len(th)):
            d = np.zeros_like(th); d[j] = h
            J[:, j] = (forward(th + d) - f0) / h
        return np.linalg.pinv(J.T @ J + 1e-6 * np.eye(len(th))) @ J.T @ r

    # ---------- 推理 ----------
    def step(self, th, r, h_ctx=1.0):
        x = np.concatenate([th, r, [h_ctx]])
        y = self.net.forward((x - self._x_mu) / self._x_sd) * self._y_sd + self._y_mu
        dth, c = y[0, :-1], float(np.clip(y[0, -1], 0, 1))
        return dth, c

    def solve(self, forward, Y_fn, theta0, k_max=5):
        """θ_{k+1}=θ_k+α_k·c_k·TRM(θ_k,r_k)，Trust Region + Armijo + merit 单调回滚。"""
        th = np.asarray(theta0, float).copy()
        delta = self.delta * max(np.linalg.norm(th), 1.0)
        c1, rho = constants.get("armijo_c1"), constants.get("armijo_rho")
        hist = []
        for k in range(k_max):
            r = self._resid(forward, th, Y_fn)
            merit = 0.5 * float(r @ r)
            hist.append(merit)
            if np.sqrt(2 * merit) / (np.linalg.norm(Y_fn()) + 1e-30) < self.eps:
                break
            dth, c = self.step(th, r)
            if np.linalg.norm(dth) > delta:               # Trust Region 截断
                dth *= delta / np.linalg.norm(dth)
            alpha = 1.0
            for _ in range(20):                           # Armijo 回溯
                th_new = th + alpha * c * dth
                r_new = self._resid(forward, th_new, Y_fn)
                if 0.5 * r_new @ r_new <= merit - c1 * alpha * float(dth @ dth):
                    break
                alpha *= rho
            else:
                break                                       # 回滚：merit 不降放弃本步
            th = th_new
            delta = min(delta * 1.5, 0.5 * max(np.linalg.norm(th), 1.0))
        return th, hist

if __name__ == "__main__":
    # 非线性正演：Y = [exp(θ0)+θ1², θ0·θ1, sin(θ1)]，θ*=(0.5,-0.8)
    TH = np.array([0.5, -0.8])
    fwd = lambda th: np.array([np.exp(th[0]) + th[1] ** 2, th[0] * th[1], np.sin(th[1])])
    Yf = lambda: fwd(TH)
    trm = TrmSolver(2, 3)
    h = trm.train_teacher(fwd, TH, Yf)
    assert h[-1] < h[0]
    th, hist = trm.solve(fwd, Yf, theta0=np.array([1.0, -1.5]), k_max=8)
    assert hist[-1] < hist[0] and np.all(np.diff(hist) <= 1e-9), hist   # merit 单调
    err = np.linalg.norm(th - TH) / np.linalg.norm(TH)
    assert err < 0.05, (th, err)
    print(f"trm OK: train {h[0]:.4f}->{h[-1]:.5f}, solve err {err:.4f}, merit {[f'{m:.2e}' for m in hist]}")
