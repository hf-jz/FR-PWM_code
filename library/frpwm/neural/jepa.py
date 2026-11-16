"""Plasma-JEPA（报告 11.3）：隐空间预测而非重建。
z_t=E_θ(Y_t)，ẑ_{t+Δ}=P_ψ(z_t,u_t,Δ)，L = L_JEPA + λ_s·L_state（状态监督，岭回归解码头）。"""
import numpy as np
from ..nn import MLP

class PlasmaJepa:
    def __init__(self, d_y, d_z=32, d_state=6, seed=0):
        self.enc = MLP(d_y, [128, 64], d_z, seed=seed)          # E_θ
        self.pred = MLP(d_z + 2, [64, 64], d_z, seed=seed + 1)  # P_ψ(z,u_eff,Δ)
        self.d_z = d_z
        self.dec = None                                          # 状态解码 z→X_fast 摘要（岭回归）
        self.lam_s = 0.1
    def encode(self, Y):
        z = self.enc.forward(Y)
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)   # L2 归一化（报告 17 模块 2）
    def predict(self, z, u_eff=0.0, delta=1.0):
        inp = np.concatenate([np.atleast_2d(z), np.full((len(np.atleast_2d(z)), 1), u_eff),
                              np.full((len(np.atleast_2d(z)), 1), delta)], -1)
        return self.pred.forward(inp)
    def train(self, seqs, epochs=60):
        """seqs: [(Y_t, Y_{t+1})] 批。L_JEPA=‖ẑ−sg(z_{t+1})‖²；训练后拟合状态解码头。
        编码器冻结于随机初始化（随机特征）：无 target encoder 的双边更新会坍缩/漂移，
        与报告讨论的表征坍缩一致。ponytail: EMA target encoder + LeWM 高斯正则为升级路径。"""
        Y0 = np.array([s[0] for s in seqs]); Y1 = np.array([s[1] for s in seqs])
        hist = []
        for ep in range(epochs):
            z0 = self.encode(Y0)
            z1_sg = self.encode(Y1)                     # stop-gradient：不回传（lazy 正确语义）
            pred = self.pred.forward(np.concatenate([z0, np.zeros((len(z0), 2))], -1))
            loss = float(((pred - z1_sg) ** 2).mean())
            g = 2 * (pred - z1_sg) / pred.size
            self.pred.backward(g)                       # 只训预测器
            hist.append(loss)
        return hist
    def fit_state_decoder(self, Z, S):
        """状态监督：z→物理量摘要 [Te0,ne0,Zeff,Prad,beta,mhd]，岭回归闭式解。"""
        A = np.concatenate([Z, np.ones((len(Z), 1))], -1)
        W, *_ = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ S, rcond=None)
        self.dec = W
    def decode_state(self, z):
        return np.atleast_2d(z) @ self.dec[:-1] + self.dec[-1]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 合成序列：Y = [10 维观测]，时序有结构（阻尼振荡）
    t = np.linspace(0, 20, 400)
    base = np.stack([np.sin(t + i) * np.exp(-t / 15) + 0.1 * i for i in range(10)], -1)
    Y = base + 0.02 * rng.standard_normal(base.shape)
    seqs = [(Y[i], Y[i + 1]) for i in range(len(Y) - 1)]
    j = PlasmaJepa(10)
    h = j.train(seqs, epochs=150)
    assert h[-1] < 0.3 * h[0], (h[0], h[-1])
    z = j.encode(Y[:5]); assert z.shape == (5, 32) and abs(np.linalg.norm(z[0]) - 1) < 1e-4
    S = np.stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4], Y[:, 5]], -1)
    j.fit_state_decoder(j.encode(Y), S)
    err = np.abs(j.decode_state(j.encode(Y[:20])) - S[:20]).mean()
    assert err < 0.08, err
    print(f"jepa OK: L {h[0]:.4f}->{h[-1]:.5f}, decode err {err:.4f}")
