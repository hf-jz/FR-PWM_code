"""风险预测（事件级）：R = {P_event, TTE, severity, uncertainty}。"""
import numpy as np
from ..schema import RiskEvent

class RiskMonitor:
    def infer(self, state, regime):
        """MVP 规则版：由状态斜率/裕度给破坏概率与到事件时间。"""
        mhd = abs(state.get("mhd_amp", 0.0)); db = max(state.get("d2e_dt", 0.0), 0.0)
        rad = state.get("rad_frac", 0.3); beta = state.get("beta_N", 1.0)
        p_dis = float(np.clip(0.5 * regime.get("locked", 0) + 3 * mhd + 0.2 * max(beta - 2.5, 0), 0, 1))
        tte = float(np.clip(0.5 / max(p_dis, 1e-3), 0.01, 5.0))     # 概率越高到事件越近（量级模型）
        p_rad = float(np.clip(2.0 * max(rad - 0.7, 0.0) + db, 0, 1))
        unc = float(np.clip(0.05 + 0.3 * state.get("ood", 0.0), 0, 1))
        return [
            RiskEvent(P=p_dis, tte=tte, severity=1.0, unc=unc, name="disruption"),
            RiskEvent(P=p_rad, tte=float(np.clip(1.0 / max(p_rad, 1e-3), 0.05, 10.0)),
                      severity=0.6, unc=unc, name="radiation_limit"),
        ]

if __name__ == "__main__":
    rm = RiskMonitor()
    r = rm.infer(dict(mhd_amp=0.09, beta_N=2.8, rad_frac=0.8, ood=0.0), dict(locked=0.5))
    assert r[0].P > 0.3 and r[0].tte < 2.0 and r[0].unc < 0.1
    print("risk OK:", [(x.name, round(x.P, 2), round(x.tte, 2)) for x in r])
