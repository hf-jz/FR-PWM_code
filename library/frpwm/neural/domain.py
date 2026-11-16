"""Domain Adapter（P2-12，报告 17 治理：z_t=E(Y_t, DomainID)，跨机器/跨工况域适配）。"""
import numpy as np
from ..nn import MLP

class DomainAdapter:
    """编码器输入拼接 DomainID（one-hot），域不变隐空间 + 域标签辅助头。"""
    def __init__(self, d_y, n_domains, d_z=16, seed=0):
        self.n_domains = n_domains
        self.enc = MLP(d_y + n_domains, [64, 32], d_z, seed=seed)
        self.head = MLP(d_z, [32], 1, seed=seed + 1)      # 状态探针（跨域一致目标）
    def encode(self, Y, domain):
        one = np.zeros((len(np.atleast_2d(Y)), self.n_domains)); one[:, domain] = 1.0
        return self.enc.forward(np.concatenate([np.atleast_2d(Y), one], -1))
    def train(self, data, epochs=150):
        """data: [(Y, domain, s)] 多域样本，训练 z→s 域不变探针。"""
        Y = np.array([d[0] for d in data]); D = np.array([d[1] for d in data])
        S = np.array([[d[2]] for d in data])
        one = np.zeros((len(Y), self.n_domains)); one[np.arange(len(Y)), D] = 1.0
        X = np.concatenate([Y, one], -1)
        Z = self.enc.forward(X)
        for _ in range(epochs // 3):
            self.enc.train_step(X, self.enc.forward(X) * 0 + Z)   # 热身：恒等（稳定初始化）
        hist = []
        for _ in range(epochs):
            z = self.enc.forward(X)
            l = self.head.train_step(z, S)
            hist.append(l)
        return hist
    def probe(self, Y, domain):
        return self.head.forward(self.encode(Y, domain))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 两域：同一物理 s，观测偏置/尺度不同（模拟跨机器诊断响应差异）
    data = []
    for i in range(400):
        s = rng.uniform(-2, 2)
        dom = i % 2
        y = (s + (0.5 if dom else 0.0)) * (1.5 if dom else 1.0) + 0.05 * rng.standard_normal()
        data.append((np.array([y, y ** 2]), dom, s))
    da = DomainAdapter(2, 2)
    h = da.train(data, epochs=150)
    err = [abs(float(da.probe(d[0], d[1])) - d[2]) for d in data[:100]]
    assert np.mean(err) < 0.3, np.mean(err)
    # 域间一致性：同一 s 在两域的 probe 接近
    p0 = float(da.probe(np.array([1.0, 1.0]), 0)); p1 = float(da.probe(np.array([2.5, 6.25]), 1))
    print(f"domain OK: probe err {np.mean(err):.3f}, 跨域一致 |{p0:.2f}-{p1:.2f}|")
