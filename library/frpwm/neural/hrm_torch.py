"""HRM 正式版（PyTorch BPTT，报告 11.4）：双时间尺度 GRU → multi-label regime 概率。
hL 每步更新（快 GRU cell），hH 每 K 步更新（慢 GRU cell）；[hL,hH] → 6 标签 sigmoid。
HRM 唯一职责 = Regime Reasoning（M_t 概率 + prior），精修归 TRM。"""
import numpy as np
import torch
import torch.nn as nn

LABELS = ["H-mode", "ELM", "locked", "tearing", "impurity", "ramp-down"]

class HrmTorch(nn.Module):
    def __init__(self, d_in, d_h=32, K=5):
        super().__init__()
        self.K = K
        self.gL = nn.GRUCell(d_in, d_h)
        self.gH = nn.GRUCell(d_h, d_h)
        # 输出看 [x_t 直通, hL, hH]：长序列下 GRU 隐态会洗掉瞬时判别信息（实测 hL 在
        # elm/正常两段几乎相同），直通支路保住当前观测，递归状态提供时序上下文。
        self.out = nn.Linear(d_in + 2 * d_h, len(LABELS))
    def forward(self, x_seq):
        """x_seq: (B,T,d_in) → 每步 multi-label 概率 (B,T,6)。hH 每 K 步刷新。"""
        B, T, _ = x_seq.shape
        hL = torch.zeros(B, self.gL.hidden_size, device=x_seq.device)
        hH = torch.zeros(B, self.gH.hidden_size, device=x_seq.device)
        outs = []
        for t in range(T):
            hL = self.gL(x_seq[:, t], hL)
            if t % self.K == 0:
                hH = self.gH(hL, hH)
            outs.append(self.out(torch.cat([x_seq[:, t], hL, hH], -1)))
        return torch.sigmoid(torch.stack(outs, 1))

class HrmTrainer:
    def __init__(self, d_in, d_h=32, K=5, lr=5e-3, device="cpu", pos_weight=3.0):
        self.dev = device
        self.model = HrmTorch(d_in, d_h, K).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.pw = pos_weight     # 事件步占比低（~25%），不加权会退化成全判正常
    def train(self, X, Y, epochs=40, bs=32):
        X = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.dev)
        Y = torch.tensor(np.asarray(Y), dtype=torch.float32, device=self.dev)
        n = len(X); hist = []
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.dev)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                p = self.model(X[idx]).clamp(1e-6, 1 - 1e-6)
                y = Y[idx]
                loss = -(self.pw * y * p.log() + (1 - y) * (1 - p).log()).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                tot += float(loss)
            hist.append(tot / max(1, n // bs))
        return hist
    @torch.no_grad()
    def infer(self, x_seq):
        """x_seq: (T,d_in) 单序列 → 每步概率 (T,6)。"""
        X = torch.tensor(np.asarray(x_seq, np.float32)[None], device=self.dev)
        return self.model(X)[0].cpu().numpy()

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xs, Ys = [], []
    for s in range(80):
        T = 60
        elm = np.zeros(T); elm[20:35] = 1.0
        x = np.stack([0.005 + 0.115 * elm + 0.02 * rng.standard_normal(T),
                      0.3 + 0.4 * elm + 0.05 * rng.standard_normal(T),
                      0.02 * rng.standard_normal(T)], -1)
        y = np.zeros((T, len(LABELS)))
        y[:, 0] = 1 - elm; y[:, 1] = elm
        Xs.append(x); Ys.append(y)
    tr = HrmTrainer(3, K=5)
    h = tr.train(np.array(Xs), np.array(Ys), epochs=150)
    p = tr.infer(Xs[0])
    acc = ((p[:, 1] > 0.5) == Ys[0][:, 1]).mean()
    assert h[-1] < 0.5 * h[0] and acc > 0.9, (h[0], h[-1], acc)
    print(f"hrm_torch OK: L {h[0]:.4f}->{h[-1]:.4f}, ELM 逐步准确率 {acc:.3f}")
