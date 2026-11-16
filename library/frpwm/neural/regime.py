# -*- coding: utf-8 -*-
"""Regime Reasoning Module（RRM）——FR-PWM 2.2 功能接口冻结（plan_eval09131124）。

冻结的是功能而非网络结构：
    必须存在 Regime State M_t 与 Regime Reasoning 功能
        M_t = R(z_{t-W:t}, X_fast_{t-W:t})   （multi-label regime 概率 + prior）
    默认实现 = HRM（Hierarchical Reasoning Model，H/L 两级递归）
    允许未来替换：SSM / HMM / Temporal Transformer / rule-based regime detector
    （替换实现只需满足本模块的 RegimeReasoning 接口契约，不得改变 M_t 语义）

定位（专家§13/§18）：HRM 是"判态"（Regime Prior / Context Reasoner），可弱化可替换；
TRM 才是"解态"核心。两者均为 UPSSE 内部组件，不得成为 Layer C 之外的第二个世界模型。
调度：跨周期递归 HRM(t)→TRM(t)→HRM(t+1)，N_outer=1（ch11 已冻结）。

注意：默认实现 HRMDefault 的**接口形态冻结**；权重未经工况标签训练（M_t 标签需装置
运行数据库，属 Device Freeze 阶段数据），当前仅承载接口契约——生产精度不作声明。
"""
from abc import ABC, abstractmethod

REGIME_LABELS = ("L-mode", "H-mode", "ELM", "locked", "tearing",
                 "impurity_accumulation", "ramp_up", "flat_top", "ramp_down", "OOD")


class RegimeReasoning(ABC):
    """Regime Reasoning 功能接口（冻结契约）。

    infer(z_hist, xfast_hist) -> M_t
      z_hist:     (B, W, dz)   隐表示历史窗
      xfast_hist: (B, W, dx)   X^fast 历史窗
      返回 M_t:   (B, len(REGIME_LABELS)) 概率向量（每维 ∈[0,1]，multi-label sigmoid 语义）
    """

    labels = REGIME_LABELS

    @abstractmethod
    def infer(self, z_hist, xfast_hist):
        ...

    def check_contract(self, z_hist, xfast_hist):
        """接口契约自检：形状/标签数/概率域。"""
        import numpy as np
        m = self.infer(z_hist, xfast_hist)
        m = np.asarray(m, dtype=float)
        assert m.ndim == 2 and m.shape[0] == len(z_hist), f"M_t batch 维错误 {m.shape}"
        assert m.shape[1] == len(self.labels), f"M_t 维度 {m.shape[1]} != {len(self.labels)}"
        assert (m >= 0).all() and (m <= 1).all(), "M_t 越出 [0,1] 概率域"
        return m


class HRMDefault(RegimeReasoning):
    """默认实现：HRM 两级递归（H 慢态每 K 步更新、L 快态每步更新）。

    torch 可用时走网络前向；无 torch 时退化为零先验（接口契约保持）。
    """

    PROFILE_ID = "RegimeReasoningProfile-HRM-v1"   # plan_eval09140906 实现 profile 冻结命名

    def __init__(self, dz=96, dx=8, h=64, K=4):
        self.dz, self.dx, self.h, self.K = dz, dx, h, K
        try:
            import torch
            self._torch = torch
            self.fL = torch.nn.GRUCell(dz + dx, h)
            self.fH = torch.nn.GRUCell(h, h)
            self.head = torch.nn.Linear(h, len(self.labels))
        except ImportError:
            self._torch = None

    def infer(self, z_hist, xfast_hist):
        import numpy as np
        B, W, _ = z_hist.shape
        if self._torch is None:
            return np.full((B, len(self.labels)), 0.5)   # 无 torch：均匀先验占位
        torch = self._torch
        with torch.no_grad():
            z = torch.as_tensor(z_hist, dtype=torch.float32)
            x = torch.as_tensor(xfast_hist, dtype=torch.float32)
            hL = torch.zeros(B, self.h)
            hH = torch.zeros(B, self.h)
            for t in range(W):
                hL = self.fL(torch.cat([z[:, t], x[:, t]], 1), hL)
                if (t + 1) % self.K == 0 or t == W - 1:
                    hH = self.fH(hL, hH)                 # H 层每 K 步/末步更新
            return torch.sigmoid(self.head(hH)).numpy()


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(7)
    zh = rng.standard_normal((3, 12, 96))
    xh = rng.standard_normal((3, 12, 8))
    for impl in (HRMDefault(),):
        m = impl.check_contract(zh, xh)
        assert set(impl.labels) == set(REGIME_LABELS)
        print(f"PASS regime contract: {type(impl).__name__} M_t{m.shape} "
              f"range[{m.min():.2f},{m.max():.2f}]")
