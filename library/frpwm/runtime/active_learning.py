"""主动学习环（P2-14，报告 11 章：不确定度 → Virtual Plasma 采样 → 物理验证 → 重训）。"""
import numpy as np

class ActiveLearner:
    def __init__(self, forward, theta_lo, theta_hi):
        self.forward = forward          # Virtual Plasma 正演 θ→Y
        self.lo, self.hi = theta_lo, theta_hi
        self.buffer = []
    def uncertainty_map(self, model_var_fn, grid):
        """网格不确定度：model_var_fn(θ)→σ²（如 RF 集成方差）。"""
        return np.array([model_var_fn(th) for th in grid])
    def propose(self, model_var_fn, n=5):
        """选最能降不确定度的候选点（最大 σ² 区域）。"""
        grid = [np.array(x) for x in
                np.array(np.meshgrid(*[np.linspace(self.lo[i], self.hi[i], 8)
                                       for i in range(len(self.lo))])).T.reshape(-1, len(self.lo))]
        var = self.uncertainty_map(model_var_fn, grid)
        idx = np.argsort(var)[-n:]
        return [grid[i] for i in idx]
    def acquire(self, model_var_fn, n=5):
        """采样（Virtual Plasma 正演）→ 物理验证（有限/非负）→ 入缓冲。"""
        pts = self.propose(model_var_fn, n)
        got = 0
        for th in pts:
            y = self.forward(th)
            if np.all(np.isfinite(y)) and np.all(y >= -1e30):
                self.buffer.append((th, y)); got += 1
        return got
    def cycle(self, model, retrain, var_fn, n=5):
        """一轮：σ 前 → 采样 → 重训 → σ 后（应下降）。"""
        grid_probe = [np.array(x) for x in
                      np.array(np.meshgrid(*[np.linspace(self.lo[i], self.hi[i], 6)
                                             for i in range(len(self.lo))])).T.reshape(-1, len(self.lo))]
        before = float(np.mean([var_fn(th) for th in grid_probe]))
        self.acquire(var_fn, n)
        retrain(self.buffer)
        after = float(np.mean([var_fn(th) for th in grid_probe]))
        return before, after

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 学习问题：y = θ²（1 维），模型初期只在 θ∈[0,0.4] 有数据
    al = ActiveLearner(lambda th: np.array([th[0] ** 2]), np.array([0.0]), np.array([1.0]))
    samples = [(np.array([x]), np.array([x ** 2])) for x in np.linspace(0, 0.4, 6)]
    al.buffer.extend(samples)
    class Toy:
        def __init__(self): self.W = None
        def retrain(self, buf):
            X = np.array([b[0] for b in buf]); Y = np.array([b[1] for b in buf])
            A = np.concatenate([X, np.ones_like(X)], -1)
            self.W = np.linalg.lstsq(A.T @ A + 1e-6 * np.eye(2), A.T @ Y, rcond=None)[0]
        def var(self, th):
            # 简模型方差：远离训练区则大
            d = min(abs(th[0] - b[0][0]) for b in al.buffer)
            return d ** 2
    toy = Toy(); toy.retrain(al.buffer)
    b, a = al.cycle(toy, toy.retrain, toy.var, n=4)
    assert a < b, (b, a)
    print(f"active_learning OK: 平均不确定度 {b:.4f} -> {a:.4f}")
