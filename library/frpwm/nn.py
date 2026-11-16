"""共享微型 MLP（numpy 手写反向传播 + Adam）——JEPA/TRM/残差网络复用，禁止三份拷贝。
够用即可：2 隐层 tanh，回归 MSE。ponytail: 无批量层/dropout，需要时上 PyTorch。"""
import numpy as np

class MLP:
    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        hs = [d_in] + list(d_hidden) + [d_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(hs[:-1], hs[1:])]
        self.b = [np.zeros(b) for b in hs[1:]]
        self.opt = Adam(self)

    def _flat(self):
        return np.concatenate([w.ravel() for w in self.W] + [b.ravel() for b in self.b])

    def forward(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        self._a = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self._a[-1] @ W + b
            self._a.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return self._a[-1]

    def backward(self, grad_out):
        g = np.atleast_2d(grad_out)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = self._a[i].T @ g / len(g)
            gb[i] = g.mean(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1 - self._a[i] ** 2)   # tanh'
        self.opt.step(gW, gb)

    def train_step(self, X, Y):
        pred = self.forward(X)
        loss = float(((pred - Y) ** 2).mean())
        self.backward(2 * (pred - Y) / pred.size)
        return loss

class Adam:
    def __init__(self, net, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.mW = [np.zeros_like(w) for w in net.W]; self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]; self.vb = [np.zeros_like(b) for b in net.b]
        self.t = 0
    def step(self, gW, gb):
        self.t += 1
        for i in range(len(self.net.W)):
            for m, v, g, p in ((self.mW, self.vW, gW, self.net.W), (self.mb, self.vb, gb, self.net.b)):
                m[i] = self.b1 * m[i] + (1 - self.b1) * g[i]
                v[i] = self.b2 * v[i] + (1 - self.b2) * g[i] ** 2
                mh = m[i] / (1 - self.b1 ** self.t); vh = v[i] / (1 - self.b2 ** self.t)
                p[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 5)); Y = np.sin(X[:, :1]) + 0.5 * X[:, 1:2]
    net = MLP(5, [32, 32], 1, seed=1)
    l0 = net.train_step(X, Y)
    for _ in range(300):
        l = net.train_step(X, Y)
    assert l < 0.05 * l0, (l0, l)
    pred = net.forward(rng.normal(0, 1, (50, 5)))
    assert pred.shape == (50, 1)
    print(f"mlp OK: {l0:.4f} -> {l:.5f}")
