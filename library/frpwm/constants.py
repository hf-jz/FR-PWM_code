"""FR-PWM 2.1.3 Implementation Freeze（附录 A 的代码形式，Freeze v1.3）。
改值走版本治理，不得在调用处散落字面量。"""
REGISTRY = {
    "schema_version": 4,            # u32，Schema v1.4
    "artifact_version": "frpwm-2.2",
    "eps_resid": 0.01,          # TRM/反演残差收敛阈 ‖r‖/‖Y‖
    "trust_delta0": 0.1,        # Trust Region 初始半径 (×‖θ0‖)
    "armijo_c1": 1e-4, "armijo_rho": 0.5,
    "gate_a": 1.0, "gate_b": 2.0, "gate_c": 1.0, "gate_d": 1.0,   # Fast Observer 门控 (归一化后)
    "delta_w_max": 0.2,         # 门控变化率限制 /cycle
    "sigma0": {"Te": 50.0, "ne": 1e18, "Zeff": 0.1},   # 门控逐状态量归一化基准 (2.1.2)
    "q_id_max": 0.5,            # I^ident=2 时 Q_t 上限
    "q_min": 0.3,               # ControlEligibility 质量阈
    "eps_horizon": 1e-3,        # Joint Chance horizon 总预算 (Σ_{k,j} ε_{j,k} ≤ ε_horizon)
    "eps_basis": 1e-3,          # 基函数截断能标
    "eps_experiment": 1e-3,     # 主动实验风险阈
    "age_max": {"spec": 0.020, "mag": 0.001, "thomson": 0.100, "efit": 0.050},  # s
    "n_hyst": 5,                # OOD 滞回 cycle 数
    "ood_weights": {"knn": 0.2, "mah": 0.2, "ens": 0.2, "rec": 0.2, "phys": 0.2},
    "picp90_target": ("ge", 0.90),   # 单侧覆盖保证 PICP90 ≥ 0.90 − ε_stat (Clopper-Pearson)
    "eps_H": {5: 0.02, 10: 0.05, 20: 0.10, 50: 0.20},
    "corr_ds_max": 0.9,
    "zeff_rmse_max": 0.1,
    "ne_te_rmse_max": 0.10,
    "ne_profile_rel_max": 0.10,   # Z_eff^LOS 绝对发布门的 n_e 剖面相对不确定度上限（3.1；超过只能发布相对 Z_eff）
    "recall_min": 0.90, "fa_budget": (0.001, 0.01),
    "convex_coverage_min": 0.95,
    "p99_critical_ms": 10.0, "p99_slow_ms": 100.0, "p99_shield_ms": 1.0,
    "t_trm_budget_ms": 70.0, "t_physics_ms": 20.0, "t_margin_ms": 10.0,
    "t_trm_wcet_ms": 10.0, "t_wm_budget_ms": 10.0, "t_mpc_budget_ms": 5.0,
    "t_safety_budget_ms": 1.0, "t_fast_deadline_ms": 10.0, "t_slow_deadline_ms": 100.0,
    "shape_zeta": 1.51,         # Z_eff 形状因子（2.1.3：历史标定先验，非冻结常数；正演按剖面/几何/仪器响应显式计算）
    "c_miss_over_fa": (1e2, 1e3),
    # ---- 2.1.3 新增 ----
    "b_r": 3.0,                 # 神经残差可信域半径 ‖R_φ‖_{Σ⁻¹}≤B_R(Q,OOD,M)，OOD 满档→0（SAFETY 只许收紧）
    "inv_tol": {"charge": 1e-6, "particle": 1e-4, "boundary": 1e-4,
                "source": 1e-3, "adas": 1e-3},   # 输运不变量数值容差（三级处置：<ε 放行 / <10ε warning / ≥10ε 回滚）
    "e_sigma_k": 3.0,           # 代理误差双指标 E_σ 阈（E_rel<1% 且 E_σ<k，分域）
    "cbf_gamma": 1.0,           # discrete-time CBF 保持率 γ=λΔt_ctrl（默认 γ=1 最保守保持）
    "conf_n_min": 200,          # Conformal Safety Module 分层样本量下限（shot 级）
    "conf_gamma_pool": 1.25,    # 分层不足回退上一层级的保守膨胀因子
    "age_max_M": 0.100,         # regime M_t freshness 阈（=HRM 更新周期一档，s）
    "h_mpc": (5, 20),           # 控制时域 H_MPC∈[5,20]
    "h_wm": (5, 10, 20, 50),    # 世界模型验收时域 H_WM∈{5,10,20,50}
    # ---- 2.1.3 Final Verification Patch（plan_eval09102105）----
    "eps_stat": {"50": 0.10, "200": 0.06, "500": 0.04, "1000": 0.03},  # PICP 验收统计余量（分阶段收紧）
    "delta_cover": 0.05,        # 覆盖置信水平 P(PICP90≥0.90)≥1−δ
    "n_cal": 200,               # conformal 标定集下限（shot 级，分层每层同 conf_n_min）
    "n_test": 50,               # conformal 测试集下限（shot 级，首炮预注册起评）
    "pcs_fallback_default": "SafePreset",   # 未登记执行器默认丢帧策略
    # ---- FEC-01（Final Engineering Closure：CBF 非线性余项 + Trust Region）----
    "eps_lin_max": 0.05,        # 局部线性化余项上界 ‖e_lin‖_W 超阈 → 禁止 QP（走 Usafe_cert/SafePreset）
    "tr_u": 0.5,                # Trust Region 动作半径 ‖u−u_ref‖_Wu（DeviceConfig 标定，禁代码静默改）
    "tr_x": 2.0,                # Trust Region 状态半径 ‖x−x_ref‖_Wx
    "q_weights": {"meas": 1/6, "fresh": 1/6, "obs": 1/6,
                  "phys": 1/6, "ood": 1/6, "ident": 1/6},   # Q=∏q_i^{w_i} 权重（Σ=1，shadow 标定）
}
def get(k, default=None):
    return REGISTRY.get(k, default)

# 参数五分类（附录 A 冻结）：FIXED 理论/数学 · CALIBRATED 标定 · DEVICE 装置/时序/算力 · SAFETY 安全关键(只许收紧) · RESEARCH 研究可调
CATEGORIES = {
    "FIXED": ["schema_version", "artifact_version", "eps_basis", "inv_tol"],
    "CALIBRATED": ["gate_a", "gate_b", "gate_c", "gate_d", "delta_w_max", "sigma0",
                   "n_hyst", "ood_weights", "shape_zeta", "q_weights"],
    "DEVICE": ["age_max", "age_max_M", "p99_critical_ms", "p99_slow_ms", "p99_shield_ms",
               "t_trm_budget_ms", "t_physics_ms", "t_margin_ms", "t_trm_wcet_ms",
               "t_wm_budget_ms", "t_mpc_budget_ms", "t_safety_budget_ms",
               "t_fast_deadline_ms", "t_slow_deadline_ms", "ne_profile_rel_max"],
    "SAFETY": ["q_id_max", "q_min", "eps_horizon", "eps_experiment", "picp90_target",
               "b_r", "cbf_gamma", "conf_n_min", "conf_gamma_pool",
               "eps_stat", "delta_cover", "n_cal", "n_test", "pcs_fallback_default",
               "eps_lin_max", "tr_u", "tr_x"],
    "RESEARCH": ["eps_resid", "trust_delta0", "armijo_c1", "armijo_rho", "eps_H",
                 "corr_ds_max", "zeff_rmse_max", "ne_te_rmse_max", "recall_min",
                 "fa_budget", "convex_coverage_min", "c_miss_over_fa", "e_sigma_k",
                 "h_mpc", "h_wm"],
}
assert sorted(k for v in CATEGORIES.values() for k in v) == sorted(REGISTRY), "五分类必须不重不漏覆盖 REGISTRY 全键"

def category_of(k):
    """键 → 五分类之一（治理：SAFETY 类只许收紧，RESEARCH 类 shadow 标定后才晋级）。"""
    for cat, keys in CATEGORIES.items():
        if k in keys:
            return cat
    raise KeyError(k)
