# -*- coding: utf-8 -*-
"""P0-4 Diagnostic Forward Operator 注册表（plan_eval09151707 §7 / M-02 唯一规范形式）：
diag_id → H(x)。x=PlasmaState 最小集 {n_e(ρ): (n_flux,)}；复用冻结算子（Park 矩阵），
TS 类=磁面 bin 直采（H=I），L_obs=Σw_i‖y_i−H_i(x)‖² 加权残差。"""
import numpy as np


def h_point(x, A):
    """ŷ_int = A·n_e（∫n_e dl 的磁面 bin 离散——point.point_forward 同算子）。"""
    return A @ x["n_e"]


def h_ts(x, bins):
    """ŷ_ts = n_e(ρ_j) 在测量 bin 上的直采（ Thomson 类=局部测量，H=选择矩阵）。"""
    return x["n_e"][np.asarray(bins)]


def h_magnetic(x, psi_probe):
    """ŷ_mag：平衡探针最小占位——返回磁轴附近 n_e 加权的代理量（Device Freeze 时换
    真实 Green 函数 H_mag(ψ,j)，接口签名不变）。"""
    return np.array([x["n_e"][:3].mean() * psi_probe])


OPERATORS = {"point": h_point, "ts": h_ts, "magnetic": h_magnetic}


def l_obs(x, observations, weights, sigma_scale=None):
    """L_obs = Σ_i w_i‖y_i−H_i(x)‖²（可选 σ 归一）。observations: {diag_id: (y, op_args)}。"""
    total, parts = 0.0, {}
    for did, (y, kwargs) in observations.items():
        r = y - OPERATORS[did](x, **kwargs)
        if sigma_scale is not None:
            r = r / max(sigma_scale, 1e-30)
        parts[did] = float(r @ r)
    for i, did in enumerate(observations):
        total += weights[i] * parts[did]
    return float(total), parts
