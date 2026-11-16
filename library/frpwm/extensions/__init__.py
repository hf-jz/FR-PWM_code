# -*- coding: utf-8 -*-
"""FR-PWM 2.2+ 扩展层（plan_eval09151707）：Feature Flag + 公共类型。
关闭全部增强时行为必须与 2.2 baseline 完全一致（验收 A）。"""
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

_CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")  # repo 根 configs/（规范树顶层）


def load_config(name="baseline"):
    """configs/<name>.yaml → dict；缺省 baseline（增强全关）。"""
    with open(os.path.join(_CFG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def enabled(cfg, ext):
    return bool(cfg.get("extensions", {}).get(ext, {}).get("enabled", False))


# ---- Temporal Alignment 五态（禁强制插值语义）----
OBSERVED, INTERPOLATED, PREDICTED, MISSING, STALE = range(5)
MASK_NAMES = ["observed", "interpolated", "predicted", "missing", "stale"]


@dataclass
class DiagnosticProfile:
    """模块 1 契约：每诊断独立登记（九字段 metadata）。"""
    diagnostic_id: str
    channels: int
    sampling_rate: float          # Hz
    sigma_rel: float              # 相对测量误差（Σ_Y cal 分量入口）
    max_age: float = 0.005        # s，超过即 stale
    geometry: dict = field(default_factory=dict)

    def metadata(self, t, q=1.0, avail=1):
        return dict(diagnostic_id=self.diagnostic_id, channel_id=None, timestamp=float(t),
                    sampling_rate=self.sampling_rate, spatial_resolution=self.channels,
                    measurement_uncertainty=self.sigma_rel, quality_score=float(q),
                    geometry=self.geometry, availability_mask=avail)


@dataclass
class DiagSample:
    """单诊断采样帧：值 + 时间戳 + 五态掩码 + 质量分。"""
    profile: DiagnosticProfile
    t: np.ndarray                 # (n,) 采样时刻
    y: np.ndarray                 # (n, channels)
    mask: np.ndarray              # (n,) 五态
    quality: float = 1.0
