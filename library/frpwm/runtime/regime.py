"""Regime 推理（MVP 规则版，HRM 的占位实现）：multi-label 概率 + 滞回。"""
import numpy as np

class RegimeMonitor:
    LABELS = ["H-mode", "ELM", "locked", "tearing", "impurity", "ramp-down"]
    def __init__(self):
        self.prev = {k: 0.0 for k in self.LABELS}; self._hold = 0
    def infer(self, state):
        """state: dict(d2e_dt, beta_N, q95, rad_frac, mhd_amp, dIp_dt) → multi-label 概率。"""
        p = {
            "H-mode": float(np.clip(2.0 * state.get("rad_frac", 0.3), 0, 1)),
            "ELM": float(np.clip(5.0 * abs(state.get("mhd_amp", 0.0)), 0, 1)),
            "locked": float(np.clip(10.0 * max(state.get("mhd_amp", 0.0) - 0.05, 0.0), 0, 1)),
            "tearing": float(np.clip(4.0 * max(state.get("mhd_amp", 0.0) - 0.02, 0.0), 0, 1)),
            "impurity": float(np.clip(2.5 * max(state.get("d2e_dt", 0.0), 0.0), 0, 1)),
            "ramp-down": float(np.clip(2.0 * max(-state.get("dIp_dt", 0.0), 0.0), 0, 1)),
        }
        # 滞回：低置信变化保持上一 regime
        for k in p:
            if abs(p[k] - self.prev[k]) < 0.1:
                p[k] = self.prev[k]
        self.prev = p
        return p

if __name__ == "__main__":
    r = RegimeMonitor()
    p1 = r.infer(dict(mhd_amp=0.09, rad_frac=0.6, d2e_dt=0.1, dIp_dt=0.0))
    assert p1["locked"] > 0.3 and p1["H-mode"] > 0.9
    p2 = r.infer(dict(mhd_amp=0.088, rad_frac=0.6, d2e_dt=0.1, dIp_dt=0.0))
    assert p2["locked"] == p1["locked"]   # 滞回（变化 <0.1 保持）
    print("regime OK:", {k: round(v, 2) for k, v in p1.items()})
