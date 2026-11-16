# -*- coding: utf-8 -*-
"""EAST 装置适配器（plan_eval09151707 §2：EAST 特殊逻辑不进核心）。
当前为占位参数（Device Freeze 时换 DeviceConfig 实测值）；把 EAST 诊断登记为
DiagnosticProfile，并把采样帧包装成统一 DiagSample。"""
import numpy as np

from ...extensions import DiagnosticProfile, DiagSample

# ponytail: 占位几何（与 point.py 自检同弦系），Device Freeze 时由 DeviceConfig 替换
EAST_DIAGNOSTICS = [
    DiagnosticProfile("point", channels=11, sampling_rate=1e4, sigma_rel=0.02,
                      geometry={"chords": "fan11"}),
    DiagnosticProfile("ts", channels=5, sampling_rate=100.0, sigma_rel=0.05,
                      geometry={"bins": [2, 6, 10, 14, 18]}),
]


def wrap_point(y_frames, t, quality=1.0):
    prof = EAST_DIAGNOSTICS[0]
    return DiagSample(prof, np.asarray(t), np.asarray(y_frames),
                      np.zeros(len(t), dtype=int), quality)


def wrap_ts(y_frames, t, quality=1.0):
    prof = EAST_DIAGNOSTICS[1]
    return DiagSample(prof, np.asarray(t), np.asarray(y_frames),
                      np.zeros(len(t), dtype=int), quality)
