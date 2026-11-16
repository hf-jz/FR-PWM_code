# -*- coding: utf-8 -*-
"""P0-3 Diagnostic Quality Engine（plan_eval09151707 §6）：
q_i(t)∈[0,1] 由 SNR/缺失率/饱和/漂移/时间有效性合成；融合权重 w_i=softmax(s_i)，
s_i=log q_i − log σ_i²（专家公式 s_i=fθ(z_i,q_i,σ_i) 的最小实现——冻结语义=质量高且
不确定度低者权重高，θ 可后换 learned 形式）。"""
import numpy as np

from . import OBSERVED, INTERPOLATED


def quality_score(y, mask, sigma_rel, drift=0.0, consistency=1.0):
    """分量：SNR=median|y|/σ、缺失率、饱和（|y| 贴近量程）、漂移、时间有效性、交叉一致性。
    q=∏分量（连乘保守，任一差则总分低——与 frpwm/quality.py 几何平均冻结式同族）。"""
    sig = max(sigma_rel, 1e-12)
    amp = np.median(np.abs(y)) + 1e-30
    snr = min(amp / (sig * amp), 100.0) / 100.0          # 归一 [0,1]，SNR≥100 记满
    missing_ratio = float((mask == 3).mean())
    q_missing = 1.0 - missing_ratio
    sat_ratio = float((np.abs(y) > 0.98 * np.abs(y).max(initial=1.0)).mean()) if y.size else 1.0
    q_sat = 1.0 - min(sat_ratio, 1.0) * 0.5
    q_drift = float(np.exp(-abs(drift)))
    valid = np.isin(mask, [OBSERVED, INTERPOLATED]).mean() if mask.size else 0.0
    q = snr * q_missing * q_sat * q_drift * float(valid) * float(np.clip(consistency, 0, 1))
    return float(np.clip(q, 0.0, 1.0))


def fusion_weights(qualities, sigmas):
    """w_i = exp(s_i)/Σexp(s_j)，s_i = log q_i − log σ_i²。全零质量时退化为均匀（不断流）。"""
    s = np.array([np.log(max(q, 1e-6)) - np.log(max(sg, 1e-12) ** 2)
                  for q, sg in zip(qualities, sigmas)])
    if not np.isfinite(s).all() or np.allclose([q for q in qualities], 0):
        return np.full(len(qualities), 1.0 / len(qualities))
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()
