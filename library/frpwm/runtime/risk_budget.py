"""Joint Chance 风险预算分配（报告 2.1.2 ①）：Σ_k Σ_j ε_{j,k} ≤ ε_horizon。
+ TRM 迭代预算 K_max=⌊T_budget/T_WCET⌋（2.1.2 P0-5）。"""
import numpy as np
from .. import constants

def allocate_epsilon(weights, eps_horizon=None):
    """weights: {(j,k): w≥0} → {(j,k): ε_{j,k}}，ε_{j,k}=ε_h·w/Σw。全零权重报错。"""
    eh = constants.get("eps_horizon", 1e-3) if eps_horizon is None else eps_horizon
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    s = sum(w.values())
    assert s > 0, "权重全零无法分配风险预算"
    alloc = {k: eh * v / s for k, v in w.items()}
    assert abs(sum(alloc.values()) - eh) < 1e-12
    return alloc

def check_budget(alloc, eps_horizon=None):
    eh = constants.get("eps_horizon", 1e-3) if eps_horizon is None else eps_horizon
    return sum(alloc.values()) <= eh + 1e-12

def joint_chance_check(p_viol, alloc=None, weights=None, eps_horizon=None):
    """联合违约概率校验：Boole 上界 Σp ≤ ε_horizon 且逐格 p_{j,k} ≤ ε_{j,k}。
    p_viol: {(j,k): P(g_j(X_{t+k})>0|I_t)}；alloc 缺省时按 weights 现算。
    → (ok, used, viol_cells)。used=Σp（Boole 上界）。"""
    eh = constants.get("eps_horizon", 1e-3) if eps_horizon is None else eps_horizon
    if alloc is None:
        alloc = allocate_epsilon(weights or {k: 1.0 for k in p_viol}, eh)
    used = float(sum(max(float(p), 0.0) for p in p_viol.values()))
    viol = [k for k, p in p_viol.items() if p > alloc.get(k, 0.0)]
    return (used <= eh and not viol), used, viol

def trm_k_max(remaining_ms, wcet_ms=None, hard_cap=20):
    """K_max=⌊remaining/T_TRM,WCET⌋（2.1.2 P0-6：按剩余 deadline 动态定步数，
    禁止困难样本无条件增至 20）。"""
    wcet = constants.get("t_trm_wcet_ms", 10.0) if wcet_ms is None else wcet_ms
    return int(np.clip(int(remaining_ms // wcet), 1, hard_cap))

# ---- Chance Constraint 概率闭合（2.1.3：Σ→P(g>0) 三层转换） ----
from scipy.stats import norm

def margin_L1(a, mu, b, sigma, eps):
    """L1 仿射约束+高斯近似：aᵀμ−b+z_{1−ε}√(aᵀΣa)≤0 的裕量 m=z√(aᵀΣa)。
    返回 (m, satisfied)：对均值点 μ 是否满足确定性等价式。"""
    a = np.asarray(a, float); mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    z = float(norm.ppf(1.0 - eps))
    var = float(a @ sigma @ a)
    m = z * np.sqrt(max(var, 0.0))
    return m, float(a @ mu - b + m) <= 0.0

def margin_L2(g_mu, q_hat):
    """L2 分布无关 conformal 风险界：g(μ̂)+q̂_{1−ε}≤0。q̂ 来自 shot 级标定（分层）。"""
    return float(g_mu) + float(q_hat) <= 0.0

def margin_L3(a, mu, b, sigma, eps):
    """L3 Cantelli 分布鲁棒兜底：κ=√((1−ε)/ε)，对任意均值-协方差已知分布成立。"""
    a = np.asarray(a, float); mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    kap = float(np.sqrt((1.0 - eps) / eps))
    var = float(a @ sigma @ a)
    m = kap * np.sqrt(max(var, 0.0))
    return m, float(a @ mu - b + m) <= 0.0

def select_layer(ood=0.0, conformal_ready=True, linear_hard=False):
    """激活规则冻结（2.1.3）：线性硬约束→L1；默认 L2；OOD↑ 或标定不足→L3。"""
    if ood > 0.5 or not conformal_ready:
        return "L3"
    if linear_hard:
        return "L1"
    return "L2"

def chance_margin(a, mu, b, sigma, eps, ood=0.0, q_hat=None, linear_hard=False):
    """统一入口：按激活层给安全裕量 m 与判定（写入 SafetyDecision 的 layer 字段）。"""
    layer = select_layer(ood=ood, conformal_ready=q_hat is not None, linear_hard=linear_hard)
    if layer == "L1":
        m, ok = margin_L1(a, mu, b, sigma, eps)
    elif layer == "L3":
        m, ok = margin_L3(a, mu, b, sigma, eps)
    else:
        m, ok = float(q_hat), margin_L2(float(np.asarray(a) @ np.asarray(mu) - b), q_hat)
    return layer, m, ok

if __name__ == "__main__":
    H, J = 20, 3
    w = {(j, k): (1.0 + j) * (1.0 + 0.1 * k) for j in range(J) for k in range(H)}
    a = allocate_epsilon(w)
    assert check_budget(a) and min(a.values()) > 0
    # 语义对照：若误用 step 语义，horizon = H·ε_step = 2%（文档警告值）
    assert abs(20 * 1e-3 - 0.02) < 1e-12
    ok, used, viol = joint_chance_check({(0, k): 1e-6 for k in range(H)}, alloc=a)
    assert ok and used < 1e-3 and not viol
    ok2, _, viol2 = joint_chance_check({(0, 0): 5e-3}, alloc=a)
    assert not ok2 and (0, 0) in viol2
    assert trm_k_max(70) == 7 and trm_k_max(5) == 1 and trm_k_max(1000) == 20
    # 2.1.3 三层转换
    a1 = np.array([1.0, 0.0]); mu1 = np.array([0.5, 0.0]); S1 = np.diag([0.01, 0.01])
    m_g, ok_g = margin_L1(a1, mu1, 1.0, S1, 1e-3)
    assert abs(m_g - 3.0902 * 0.1) < 1e-3 and ok_g                    # z=3.09
    m_c, ok_c = margin_L3(a1, mu1, 1.0, S1, 1e-3)
    assert abs(m_c - 31.6 * 0.1) < 0.1 and not ok_c                   # DRCC 更保守
    assert margin_L2(-0.2, 0.1) and not margin_L2(-0.05, 0.1)
    assert select_layer(ood=0.9) == "L3" and select_layer() == "L2" and \
        select_layer(linear_hard=True) == "L1" and select_layer(conformal_ready=False) == "L3"
    lay, m_, ok_ = chance_margin(a1, mu1, 1.0, S1, 1e-3, ood=0.0, q_hat=0.4)
    assert lay == "L2" and m_ == 0.4 and ok_                          # g_μ=−0.5，+0.4≤0 满足
    lay2, _, ok2_ = chance_margin(a1, mu1, 1.0, S1, 1e-3, ood=0.0, q_hat=0.6)
    assert lay2 == "L2" and not ok2_                                  # −0.5+0.6>0 不满足
    print(f"risk_budget OK: 60 项分配 Σε={sum(a.values()):.2e}，联合校验/K_max/三层 margin 全过")
