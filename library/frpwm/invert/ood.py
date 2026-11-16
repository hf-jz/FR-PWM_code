"""KNN-OOD（多证据之一，域内外 209× 分离已验证的方法）。"""
import numpy as np

class KnnOod:
    def __init__(self, k=5):
        self.k = k; self.ref = None; self.thr = None
    def fit(self, X_ref, thr_mult=3.0):
        """X_ref: (N,d) 训练分布。阈值 = 训练集 KNN 距离 P99 × thr_mult。"""
        self.ref = np.asarray(X_ref, float)
        d = self.dist(self.ref[:: max(1, len(self.ref) // 200)])
        self.thr = np.quantile(d, 0.99) * thr_mult
        return self
    def dist(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        d2 = ((X[:, None, :] - self.ref[None, :, :]) ** 2).sum(-1)
        return np.sqrt(np.sort(d2, -1)[:, : self.k].mean(-1))
    def score(self, X):
        """OOD 距离比：>1 即域外。"""
        return self.dist(X) / max(self.thr, 1e-30)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (500, 8))
    od = KnnOod().fit(ref)
    s_in = od.score(rng.normal(0, 1, (10, 8))).mean()
    s_out = od.score(rng.normal(8, 1, (10, 8))).mean()
    assert s_out > 5 * s_in, (s_in, s_out)
    print("ood OK:", round(s_in, 3), round(s_out, 1))
