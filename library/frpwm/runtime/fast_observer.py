"""Fast Causal Observer（Layer C 快环）：双模型融合门控（FR-PWM 2.1.2 冻结式）。
l_i = α_i − a·σ̃_i − b·ÕOD_i − c·r̃_i + d·Q_i；w_neural=sigmoid(l)，w_obs=1−w_neural；
σ̃=σ/σ_{0,i}、r̃=|r|/r_{0,i} 逐状态量归一化（量纲消除）；|Δw|<Δw_max 防 ELM 门控震荡。"""
import numpy as np
from .. import constants

def gate_weights(sigma, ood, resid, Q=1.0, keys=None, prev_w=None):
    """冻结式门控 → neural 权重（observer=1−w）。逐分量 sigmoid，变化率受限。"""
    a, b, c, d = (constants.get("gate_a", 1.0), constants.get("gate_b", 2.0),
                  constants.get("gate_c", 1.0), constants.get("gate_d", 1.0))
    sigma = np.asarray(sigma, float)
    resid = np.abs(np.asarray(resid, float))
    s0_tab = constants.get("sigma0", {"Te": 50.0, "ne": 1e18, "Zeff": 0.1})
    if keys is not None:
        s0 = np.array([s0_tab.get(k, 1.0) for k in keys])
    else:
        s0 = np.full_like(sigma, float(np.mean(list(s0_tab.values()))) if sigma.size == 1 else 1.0)
    r0 = 1.0                                    # r_{0,i} 归一化基准（相对残差，无量纲）
    l = (-a * sigma / s0 - b * float(ood) - c * resid / r0 + d * float(Q))
    # 数值稳定：l 很负时 exp(-l) 会 overflow（RuntimeWarning + inf）
    w = 1.0 / (1.0 + np.exp(-np.clip(l, -700.0, 700.0)))
    if prev_w is not None:                      # |Δw|<Δw_max 滞回
        dw = constants.get("delta_w_max", 0.2)
        w = np.asarray(prev_w, float) + np.clip(w - np.asarray(prev_w, float), -dw, dw)
    return w

def fuse(x_neural, x_obs, sigma, ood, resid, Q=1.0, keys=None, prev_w=None):
    """双模型融合 → (X_fast, w_neural, w_observer)。w∈[0,1]，w+w_obs=1 恒成立。"""
    w = gate_weights(sigma, ood, resid, Q=Q, keys=keys, prev_w=prev_w)
    w = np.clip(w, 0.0, 1.0)
    x = w * np.asarray(x_neural, float) + (1.0 - w) * np.asarray(x_obs, float)
    return x, w, 1.0 - w

if __name__ == "__main__":
    # σ 小的分量更信 neural；OOD 升 → neural 权重降；Q 升 → neural 权重升
    x, wn, wo = fuse([2.0, 2.0], [1.0, 1.0], sigma=[10.0, 90.0], ood=0.0,
                     resid=[0.0, 0.0], keys=["Te", "Te"])
    assert wn[0] > wn[1] and np.allclose(wn + wo, 1)
    w_good = gate_weights([50.0], 0.0, [0.0], Q=0.9, keys=["Te"])
    w_ood = gate_weights([50.0], 2.0, [0.0], Q=0.9, keys=["Te"])
    w_lowq = gate_weights([50.0], 0.0, [0.0], Q=0.1, keys=["Te"])
    assert w_ood[0] < w_good[0] and w_lowq[0] < w_good[0]
    w_prev = gate_weights([50.0], 5.0, [0.0], Q=0.1, keys=["Te"], prev_w=[0.9])
    assert w_prev[0] >= 0.9 - 0.2 - 1e-9          # Δw_max=0.2 限幅
    print("fast_observer OK(2.1.2):", wn.round(3), "w_ood", w_ood.round(3), "w_prev", w_prev.round(3))
