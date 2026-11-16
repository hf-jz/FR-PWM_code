"""Supervisor（Layer F，规则版）：State Residual Monitor 故障分类决策矩阵。"""
import numpy as np

class Supervisor:
    """r_y/r_p/OOD 组合 → 判定 + 建议（advisory，不进控制链）。"""
    def classify(self, r_y, r_p, ood, ident=0, sigma_ens=0.0):
        ry, rp = float(r_y), float(r_p)
        if ident == 2 or (ry < 0.5 and rp > 2.0):
            return dict(kind="unidentifiable", action="I_ident=2: 禁作高置信控制量")
        if ry > 1.0 and rp < 0.5 and ood < 1.0:
            return dict(kind="sensor_fault", action="imputed 标记 + 降信任 + 诊断维修请求")
        if ry > 1.0 and rp > 1.0 and ood > 1.0:
            return dict(kind="candidate_novel_regime",
                        action="Supervisor 假设 + 仿真 + 独立诊断验证（非直接断言新物理）")
        if sigma_ens > 1.0 and ood < 1.0:
            return dict(kind="model_divergence", action="记录 + 提高采样，不降级")
        if ood > 1.0:
            return dict(kind="ood", action="多证据滞回后五级降级")
        return dict(kind="normal", action="无")

if __name__ == "__main__":
    s = Supervisor()
    assert s.classify(2.0, 0.1, 0.3)["kind"] == "sensor_fault"
    assert s.classify(2.0, 2.0, 2.0)["kind"] == "candidate_novel_regime"
    assert s.classify(0.1, 0.1, 0.1)["kind"] == "normal"
    assert s.classify(0.1, 5.0, 0.1, ident=2)["kind"] == "unidentifiable"
    print("supervisor OK")
