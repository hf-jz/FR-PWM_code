"""Plasma-JEPA（PyTorch 正式版，报告 11.3）：EMA target encoder 防坍缩。
在线编码器 E_θ 训练，目标编码器 E_θ̄ = m·E_θ̄ + (1−m)·E_θ（stop-gradient 真源）。
L = L_JEPA + λ_s·L_state（状态监督头同时训练）。"""
import numpy as np
import torch
import torch.nn as nn

class _Enc(nn.Module):
    def __init__(self, d_y, d_z):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_y, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, d_z))
    def forward(self, y):
        return nn.functional.normalize(self.net(y), dim=-1)

class JepaTorch:
    def __init__(self, d_y, d_z=32, d_state=6, ema=0.996, lam_s=0.1, lr=3e-4, device="cpu"):
        self.dev = device
        self.enc = _Enc(d_y, d_z).to(device)
        self.tgt = _Enc(d_y, d_z).to(device)
        self.tgt.load_state_dict(self.enc.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad = False
        self.pred = nn.Sequential(nn.Linear(d_z + 2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                                  nn.Linear(64, d_z)).to(device)
        self.head = nn.Linear(d_z, d_state).to(device)          # 状态监督解码头
        self.ema, self.lam_s = ema, lam_s
        self.opt = torch.optim.AdamW(list(self.enc.parameters()) + list(self.pred.parameters())
                                     + list(self.head.parameters()), lr=lr)
    def _ema_update(self):
        with torch.no_grad():
            for p, q in zip(self.enc.parameters(), self.tgt.parameters()):
                q.mul_(self.ema).add_((1 - self.ema) * p)
    def train(self, seqs, states, epochs=30, head_epochs=30, bs=256):
        """两阶段：① JEPA+EMA 学隐空间（L_JEPA）；② 冻结编码器单训状态头（L_state）。
        联合训练会让 z 在预测目标下漂移、状态头跟着发散（实测 err 0.4→1.2）。"""
        X = torch.tensor(np.asarray(seqs), dtype=torch.float32, device=self.dev)
        S = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.dev)
        n = len(X); hist = []
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.dev)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                y0, y1 = X[idx, 0], X[idx, 1]
                z0 = self.enc(y0)
                with torch.no_grad():
                    z1 = self.tgt(y1)
                zpred = self.pred(torch.cat([z0, torch.zeros(len(z0), 2, device=self.dev)], -1))
                loss = ((zpred - z1) ** 2).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step(); self._ema_update()
                tot += float(loss)
            hist.append(tot / max(1, n // bs))
        # 阶段②：冻结 enc/pred，状态头用岭回归闭式解（z 分量~0.03，优化器 wd 会压死大权重）
        z_all = self.encode(X[:, 0].cpu().numpy())
        A = np.concatenate([z_all, np.ones((len(z_all), 1))], -1)
        S_np = S.cpu().numpy()
        W, *_ = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ S_np, rcond=None)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(W[:-1].T, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(W[-1], dtype=torch.float32))
        return hist
    @torch.no_grad()
    def encode(self, Y):
        return self.enc(torch.tensor(np.atleast_2d(np.asarray(Y, np.float32)),
                                     dtype=torch.float32, device=self.dev)).cpu().numpy()
    @torch.no_grad()
    def decode_state(self, Z):
        return self.head(torch.tensor(np.atleast_2d(np.asarray(Z, np.float32)),
                                      dtype=torch.float32, device=self.dev)).cpu().numpy()
    def latent_rank(self, Y):
        """坍缩检测：隐空间协方差谱参与率 PR=(Σs)²/Σs²。坍缩 → ≈1，健康多维 ≫1。"""
        Z = self.encode(Y)
        Z = Z - Z.mean(0)
        s = np.linalg.svd(Z, compute_uv=False)
        return float((s.sum() ** 2) / max((s ** 2).sum(), 1e-30))
    def latent_cossim(self, Y):
        """平均两两余弦相似度（参考指标；周期数据天然偏高，不作坍缩判据）。"""
        Z = self.encode(Y)
        G = Z @ Z.T
        n = len(G)
        off = (G.sum() - np.trace(G)) / max(n * (n - 1), 1)
        return float(off)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 600)
    Y = np.stack([np.sin(t + i) * np.exp(-t / 15) + 0.1 * i for i in range(10)], -1)
    Y += 0.02 * rng.standard_normal(Y.shape)
    seqs = np.stack([Y[:-1], Y[1:]], 1)
    S = (Y[:, :6] - Y[:, :6].mean(0)) / (Y[:, :6].std(0) + 1e-9)   # 状态量 z-score（与真实数据路径一致）
    S = S[:len(seqs)]                                              # 每个 (t→t+1) 对一行
    j = JepaTorch(10)
    h = j.train(seqs, S, epochs=25)
    cs = j.latent_rank(Y)
    assert h[-1] < 0.5 * h[0] and cs > 2.0, (h[0], h[-1], cs)
    err = np.abs(j.decode_state(j.encode(Y[:20])) - S[:20]).mean()
    assert err < 0.15, err
    print(f"jepa_torch OK: L {h[0]:.4f}->{h[-1]:.5f}, latent PR {cs:.1f}, decode err {err:.3f}")
