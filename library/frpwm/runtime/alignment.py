"""AlignedStateSnapshot（报告 2.1.2 ②）：跨时间尺度状态对齐。
t_s ≤ t_f ≤ t_now；慢状态 Propagate 时间前推（保持+线性趋势外推，非裸 last_good）。"""
import numpy as np

class AlignedSnapshot:
    """slow 历史（≥2 样本）驱动趋势外推；单样本退化为保持。"""
    def __init__(self, trend_clip=0.5):
        self.slow_hist = []            # [(t_s, X_slow ndarray)]
        self.clip = trend_clip         # 外推幅度上限（相对 |X|）
    def push_slow(self, t_s, x_slow):
        """登记慢状态样本。新放电（t_s 回退）→ 历史清零（时间序不变量按放电维持）。"""
        if self.slow_hist and t_s < self.slow_hist[-1][0]:
            self.slow_hist.clear()
        self.slow_hist.append((float(t_s), np.asarray(x_slow, float)))
        if len(self.slow_hist) > 8:
            self.slow_hist.pop(0)
    def propagate(self, t_f):
        """X^slow(t_f) = Propagate(X^slow(t_s), t_f − t_s)。"""
        assert self.slow_hist, "no slow state"
        (t0, x0) = self.slow_hist[-1]
        assert t0 <= t_f, f"t_s={t0} > t_f={t_f} 时间序违规"
        if len(self.slow_hist) < 2:
            return x0.copy()
        (t1, x1) = self.slow_hist[-2]
        rate = (x0 - x1) / max(t0 - t1, 1e-9)
        prop = x0 + rate * (t_f - t0)
        lim = self.clip * (np.abs(x0) + 1e-9)
        return x0 + np.clip(prop - x0, -lim, lim)   # 趋势外推但限幅
    def align(self, t_f, x_fast, M=None, sigma=None, Q=None):
        t_now = t_f
        assert all(t_s <= t_now for t_s, _ in self.slow_hist)
        return dict(t=t_f, X_fast=np.asarray(x_fast, float),
                    X_slow=self.propagate(t_f), M=M, sigma=sigma, Q=Q,
                    t_slow_src=self.slow_hist[-1][0] if self.slow_hist else None)

if __name__ == "__main__":
    a = AlignedSnapshot()
    a.push_slow(12.40, np.array([100.0, 2.0]))
    a.push_slow(12.45, np.array([110.0, 2.0]))
    x = a.propagate(12.48)                       # 趋势 +6（+10/s × 0.03s）
    assert abs(x[0] - 116.0) < 1e-9 and x[1] == 2.0
    s = a.align(12.48, np.array([1.0, 2.0]))
    assert s["t_slow_src"] == 12.45
    big = AlignedSnapshot(trend_clip=0.01)       # 限幅生效
    big.push_slow(0.0, np.array([100.0])); big.push_slow(1.0, np.array([200.0]))
    assert abs(big.propagate(2.0)[0] - 202.0) < 1e-6
    print("alignment OK: Propagate 趋势前推 + 限幅 + 时间序约束")
