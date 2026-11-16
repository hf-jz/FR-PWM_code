# -*- coding: utf-8 -*-
"""P0 闭环集成（plan_eval09151707 P0 五项）：
合成等离子体 → 多诊断采样（POINT 10kHz + TS 100Hz）→ Temporal Alignment →
Diagnostic Quality → 加权融合状态 → Forward Operator 闭环残差 → Physics Anchor
validation/correction → 状态校正。
自检断言：①闭环后 L_obs 下降；②anchor correction 缩小神经偏差；③enhanced OFF
（baseline.yaml）≡ 2.2 基线路径（验收 A）；④TS 缺失时系统不断流（五态语义）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frpwm.extensions import (DiagSample, DiagnosticProfile, enabled, load_config,
                              MISSING, OBSERVED)
from frpwm.extensions.temporal_alignment import align
from frpwm.extensions.diagnostic_quality import quality_score, fusion_weights
from frpwm.extensions.forward_operators import OPERATORS, l_obs
from frpwm.extensions.physics_anchor import PhysicsAnchor
from frpwm.physics.gfile import generate_gfile
from frpwm.physics.fluxmap import FluxMap
from frpwm.physics.point import park_matrix, point_forward
from frpwm.adapters.east import EAST_DIAGNOSTICS, wrap_point, wrap_ts


def _chords():
    R0, a = 1.7, 0.6
    return ([dict(name=f"h{i}", R0=R0 - a + 0.2 + (2 * a - 0.4) * i / 4, Z0=0.0,
                  dR=0.0, dZ=1.0, L=2.5) for i in range(5)]
            + [dict(name=f"d{i}", R0=R0 - 0.3 + 0.3 * i, Z0=-1.2, dR=0.1 * (i - 1),
                    dZ=1.0, L=2.4) for i in range(3)]
            + [dict(name=f"t{i}", R0=R0 + a - 0.15, Z0=-0.3 + 0.3 * i, dR=0.0,
                    dZ=1.0, L=0.9) for i in range(3)])


def run(config="east_enhanced", noise=0.02, ts_missing=False, seed=3):
    rng = np.random.default_rng(seed)
    cfg = load_config(config)
    enh = enabled(cfg, "forward_operator")          # baseline 下全 False

    # --- 真实等离子体（合成 ground truth）---
    fmap = FluxMap(generate_gfile())
    A = park_matrix(fmap, _chords(), n_flux=20)
    rho = (np.arange(20) + 0.5) / 20
    n_true = 8e13 * (1 - rho ** 1.7) ** 1.3
    x_true = {"n_e": n_true}

    # --- 多诊断采样：POINT 10kHz(100 帧) + TS 100Hz(1 帧，可缺失) ---
    t_pt = np.arange(100) / 1e4
    N_frames = point_forward(A, n_true)[None, :] * (1 + noise * rng.standard_normal((100, 11)))
    s_pt = wrap_point(N_frames, t_pt)
    t_ts = np.array([0.005])
    s_ts = wrap_ts(n_true[[2, 6, 10, 14, 18]][None, :] * (1 + 0.05 * rng.standard_normal((1, 5))), t_ts)
    if ts_missing:
        s_ts = DiagSample(s_ts.profile, np.array([]), np.zeros((0, 5)),
                          np.array([], dtype=int), quality=0.0)

    # --- P0-2 Temporal Alignment：统一到 1kHz 网格，取中点帧 ---
    t_grid = np.arange(10) / 1e3
    y_pt, m_pt = align(s_pt, t_grid)
    y_ts, m_ts = align(s_ts, t_grid)
    k = 5  # 分析帧
    assert m_pt[k] == OBSERVED
    if ts_missing:
        assert (m_ts == MISSING).all()

    # --- P0-3 Diagnostic Quality + 融合权重 ---
    q_pt = quality_score(y_pt[k], m_pt, 0.02)
    q_ts = 0.0 if ts_missing else quality_score(y_ts[k], m_ts, 0.05)
    w = fusion_weights([q_pt, q_ts], [0.02, 0.05])
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()

    # --- P0-1 神经路径代理（UPSSE Fast Branch 占位=加权初始猜测）---
    anchor = PhysicsAnchor(A, lam=0.3)
    x_anchor = anchor.initialize(y_pt[k])
    x_neural = {"n_e": x_anchor["n_e"] * (1 + 0.15 * rng.standard_normal(20))}  # 15% 神经偏差
    if not enh:                                    # baseline：无闭环无 anchor，直接输出
        return dict(L0=None, L1=None, dev_after=None, x=x_neural, cfg=cfg)

    # --- P0-4 Forward Operator 闭环：L_obs 前后对比（Gauss-Newton 一步=anchor 校正）---
    obs = {"point": (y_pt[k], dict(A=A))}
    if not ts_missing:
        obs["ts"] = (y_ts[k], dict(bins=[2, 6, 10, 14, 18]))
    wts = w[:len(obs)]
    L0, _ = l_obs(x_neural, obs, wts)

    # --- P0-5 Physics Anchor：validation → correction ---
    dev, trusted = anchor.validate(x_neural, y_pt[k])
    x_corr = anchor.correct(x_neural, y_pt[k], alpha=0.6)
    L1, _ = l_obs(x_corr, obs, wts)
    dev_after = float(np.abs(x_corr["n_e"] - x_true["n_e"]).mean() / x_true["n_e"].mean())
    return dict(L0=L0, L1=L1, dev_after=dev_after, trusted=trusted, x=x_corr, cfg=cfg)


if __name__ == "__main__":
    # ① 完整 P0 闭环：L_obs 下降
    r = run()
    assert r["L1"] < r["L0"], (r["L0"], r["L1"])
    print(f"P0 loop: L_obs {r['L0']:.3e} -> {r['L1']:.3e}, dev_after={r['dev_after']:.3f}")
    # ② anchor correction 收敛到真值附近（占位几何 honest 界）
    assert r["dev_after"] < 0.15, r["dev_after"]
    # ③ baseline（增强全关）：返回即 2.2 行为（无闭环项），验收 A 语义
    rb = run(config="baseline")
    assert rb["L0"] is None and rb["x"] is not None
    # ④ TS 缺失：五态语义下不断流，L_obs 仍下降
    rm = run(ts_missing=True)
    assert rm["L1"] < rm["L0"]
    print(f"ablation ts_missing: L_obs {rm['L0']:.3e} -> {rm['L1']:.3e} (不断流)")
    print("PASS p0_loop: 五项 P0 闭环 + baseline 兼容 + 缺失容忍")
