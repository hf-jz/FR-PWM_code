# -*- coding: utf-8 -*-
"""P0-5 Physics Anchor（plan_eval09151707 §14 五角色：initialization/constraint/
correction/fallback/validation）。包装经典重建链（FluxMap 平衡 + point_invert LSQ），
不是训练标签；与 FR-PWM 神经路径并行常驻，OOD/异常时接管并校验。"""
import numpy as np

from ..physics.point import point_invert


class PhysicsAnchor:
    """EFIT/经典反演锚。dev_threshold：|x_neural−x_anchor|/x_anchor 超阈→降信任（validation）。"""

    def __init__(self, A, lam=0.3, dev_threshold=0.25):
        self.A, self.lam, self.dev_threshold = A, lam, dev_threshold

    def initialize(self, y_point):
        """角色①initialization：经典 LSQ 反演给出 x0（神经路径冷启动初值）。"""
        return {"n_e": point_invert(y_point, self.A, lam=self.lam)}

    def fallback(self, y_point):
        """角色④fallback：主路径异常时系统仍保有物理状态估计（=initialize 的显式语义）。"""
        return self.initialize(y_point)

    def validate(self, x_neural, y_point):
        """角色⑤validation（②constraint/③correction 的判据）：与锚的相对偏差。
        返回 (deviation, trusted)。trusted=False → 调用方升不确定度/降信任/转 fallback。"""
        xa = self.initialize(y_point)["n_e"]
        dev = float(np.abs(x_neural["n_e"] - xa).mean() / (xa.mean() + 1e-30))
        return dev, bool(dev <= self.dev_threshold)

    def correct(self, x_neural, y_point, alpha=0.5):
        """角色③correction：向锚方向阻尼校正 x←x+α(x_anchor−x)（Recursive Diagnostic
        Inference 的最小实现；α∈(0,1] 配置化）。"""
        xa = self.initialize(y_point)["n_e"]
        return {"n_e": x_neural["n_e"] + alpha * (xa - x_neural["n_e"])}
