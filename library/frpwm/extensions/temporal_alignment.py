# -*- coding: utf-8 -*-
"""P0-2 Temporal Alignment（plan_eval09151707 §5）：统一时间轴 + 五态掩码。
禁强制插值：网格点上无样本时按最近邻取值但 mask=interpolated/stale，语义保留。"""
import numpy as np

from . import OBSERVED, INTERPOLATED, MISSING, STALE


def align(sample, t_grid):
    """DiagSample → (y_aligned (T,C), mask (T,))。
    网格点距最近采样 < dt_sample/2 → observed；< max_age → interpolated（取值但标记）；
    否则 stale；该诊断全程无样本 → missing。"""
    T, C = len(t_grid), sample.y.shape[1]
    y_out = np.zeros((T, C))
    mask = np.full(T, MISSING, dtype=int)
    if len(sample.t) == 0:
        return y_out, mask
    dt = np.median(np.diff(sample.t)) if len(sample.t) > 1 else sample.profile.max_age
    idx = np.clip(np.searchsorted(sample.t, t_grid), 0, len(sample.t) - 1)
    # 取左右更近者
    left = np.clip(idx - 1, 0, len(sample.t) - 1)
    choose_left = np.abs(sample.t[left] - t_grid) < np.abs(sample.t[idx] - t_grid)
    near = np.where(choose_left, left, idx)
    age = np.abs(sample.t[near] - t_grid)
    y_out = sample.y[near]
    mask = np.where(age <= dt / 2, OBSERVED,
                    np.where(age <= sample.profile.max_age, INTERPOLATED, STALE))
    # 采样帧自带 missing 掩码传播
    frame_bad = sample.mask[near] == MISSING
    mask[frame_bad] = MISSING
    return y_out, mask
