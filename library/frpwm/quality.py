"""Q_t：唯一状态质量分（四核心指标之一）。C_t 仅为其运维显示别名。
2.1.2：两级质量 + 动作级聚合 Q_control(u)=Aggregation(Q_i(ρ), Dep(u))。"""
import numpy as np
from . import constants

def state_quality(meas_quality=1.0, freshness=1.0, obs_disagree=0.0,
                  phys_resid=0.0, ood=0.0, ident=0):
    """Q ∈ [0,1]。生产函数冻结（2.1.3）：Q=∏q_i^{w_i}（Σw=1，加权几何平均），
    单调性天然满足（∂Q/∂uncertainty≤0 等）；任何语言/模块实现必须用此式，禁止 MLP。
    分量 q_i∈[0,1]：meas/fresh/obs=exp(−2|obs|)/phys=exp(−|r_p|)/ood=exp(−1.5·s̃_ood)/ident。"""
    w = constants.get("q_weights")
    q_ident = 1.0 if ident == 0 else (0.7 if ident == 1 else constants.get("q_id_max", 0.5))
    comp = {"meas": float(np.clip(meas_quality, 0, 1)),
            "fresh": float(np.clip(freshness, 0, 1)),
            "obs": float(np.exp(-2.0 * abs(obs_disagree))),
            "phys": float(np.exp(-1.0 * abs(phys_resid))),
            "ood": float(np.exp(-1.5 * max(ood, 0.0))),
            "ident": float(np.clip(q_ident, 0, 1))}
    w_sum = sum(w.values())
    Q = float(np.prod([comp[k] ** (w[k] / w_sum) for k in comp]))
    if ident == 2:                                  # 2.1.1 冻结：I^ident=2 ⇒ Q≤q_id_max
        Q = min(Q, constants.get("q_id_max", 0.5))
    elif ident == 1:
        Q = min(Q, 0.7)
    return float(np.clip(Q, 0, 1))

def aggregate_q(q_profiles, dep=None):
    """动作级聚合：dep={量: ρ 段 slice 或 None=全段}，取相关段最小值（保守）。
    q_profiles: {量: 逐段质量 ndarray 或标量}。dep=None 退化为全量最小。"""
    vals = []
    for k, q in q_profiles.items():
        seg = (dep or {}).get(k)
        if np.isscalar(q):
            vals.append(float(q))
        else:
            arr = np.asarray(q, float)
            vals.append(float(arr[seg].min() if seg is not None else arr.min()))
    return min(vals) if vals else 1.0

if __name__ == "__main__":
    assert state_quality() == 1.0
    q_ood = state_quality(ood=2.0)
    assert 0.0 < q_ood < 0.65, q_ood                 # 几何平均：单因子恶化不再一票否决
    assert q_ood < state_quality(ood=0.5)            # 单调性：OOD↑ ⇒ Q↓
    assert state_quality(ident=2) <= 0.5             # 2.1.1 冻结 cap 仍成立
    assert state_quality(ident=1) <= 0.7
    assert state_quality(phys_resid=2.0) < state_quality()
    qp = {"Te": np.array([0.9] * 10 + [0.2] * 10), "ne": np.full(20, 0.8)}
    assert aggregate_q(qp) == 0.2                                   # 全局最差段
    assert aggregate_q(qp, {"Te": slice(10, 20)}) == 0.2              # 边缘段
    assert aggregate_q(qp, {"Te": slice(0, 10), "ne": None}) == 0.8   # 芯部段
    print("quality OK: Q_control 动作级聚合（按 Dep(u) 取相关段最小）")
