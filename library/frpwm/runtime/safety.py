"""Safety Shield（Layer E，MVP 监视版）：ODD + 包络查询 + stale-state + 五级降级。
2.1.2：SafetyDecision 结构体 + reason_code（HIL→FMEA→Audit→RootCause 闭环）。"""
from dataclasses import dataclass, field
import numpy as np
from .. import constants

ODD = dict(beta_N=(0.2, 3.5), q95=(2.0, 12.0), zeff=(1.0, 6.0), rad_frac=(0.0, 0.95))

REASON_CODES = ("ODD_VIOLATION", "STALE_STATE", "OOD", "IDENT_FAIL",
                "CHANCE_CONSTRAINT", "SLEW_RATE", "REACHABILITY",
                "DEADLINE", "MODEL_UNCERTAINTY")

@dataclass
class SafetyDecision:
    status: str                       # approved / corrected / rejected
    reason_code: tuple = ()
    violated_constraint: tuple = ()
    risk_budget_used: float = 0.0
    state_age: float = 0.0
    odd_status: bool = True
    eligibility_status: bool = True
    fallback_level: int = 0
    correction: float = 0.0

class SafetyShield:
    def __init__(self):
        self.ood_run = 0; self.degrade = 0
    def in_odd(self, state):
        return all(lo <= state.get(k, (lo + hi) / 2) <= hi for k, (lo, hi) in ODD.items())
    def check(self, state, stale_ch, ood_score):
        """→ (allowed, degrade, reasons)。stale/OOD 连续 N 拍才降级（滞回）。"""
        reasons = []
        if not self.in_odd(state):
            reasons.append("ODD violation")
        if stale_ch:
            reasons.append(f"AgeViolation: {stale_ch}")
        n_h = constants.get("n_hyst", 5)
        self.ood_run = self.ood_run + 1 if ood_score > 1.0 else 0
        if self.ood_run >= n_h:
            self.degrade = min(self.degrade + 1, 4); reasons.append("OOD hysteresis degrade")
        elif self.degrade > 0 and self.ood_run == 0:
            self.degrade = max(self.degrade - 1, 0)
        if stale_ch:
            self.degrade = max(self.degrade, 2)   # stale state → Safety Degrade
        allowed = self.in_odd(state) and self.degrade < 4
        return allowed, self.degrade, reasons
    def check_decision(self, state, stale_ch, ood_score, state_age=0.0, ident=0):
        """→ SafetyDecision（2.1.2 结构化输出；check() 保持兼容）。"""
        allowed, degrade, reasons = self.check(state, stale_ch, ood_score)
        codes, viol = [], []
        if not self.in_odd(state):
            codes.append("ODD_VIOLATION"); viol.append("ODD")
        if stale_ch:
            codes.append("STALE_STATE"); viol.append(f"age>{stale_ch}")
        if self.ood_run > 0 and ood_score > 1.0:
            codes.append("OOD")
        if ident == 2:
            codes.append("IDENT_FAIL")
        status = "approved" if allowed and not codes else ("rejected" if not allowed else "corrected")
        return SafetyDecision(status=status, reason_code=tuple(codes),
                              violated_constraint=tuple(viol),
                              state_age=state_age, odd_status=self.in_odd(state),
                              eligibility_status=ident != 2, fallback_level=degrade)

def eligibility(ages, idents, Qs, dep=None):
    """ControlEligibility 硬门（2.1.2 P0-3）：E_i=1[I_i<2]·1[age_i<age_max,i]·1[Q_i>Q_min,i]；
    E_u=∏_{i∈Dep(u)} E_i。ages/idents/Qs: {量: 值}；age_max 取 constants.age_max（按通道映射，
    未映射量用 max(age_max) 保守档）。返回 (E: {量: 0/1}, E_u: 0/1)。"""
    age_tab = constants.get("age_max", {})
    q_min = constants.get("q_min", 0.3)
    age_def = max(age_tab.values()) if age_tab else 0.1
    E = {}
    for k in set(ages) | set(idents) | set(Qs):
        amax = age_tab.get(k, age_def)
        e = (1 if idents.get(k, 0) < 2 else 0) \
            * (1 if ages.get(k, 0.0) < amax else 0) \
            * (1 if Qs.get(k, 1.0) > q_min else 0)
        E[k] = e
    keys = dep if dep is not None else list(E)
    E_u = 1
    for k in keys:
        E_u *= E.get(k, 0)
    return E, E_u

# ================= 2.1.3：ODDState 五元组 + regime 时态一致性门 =================
@dataclass
class ODDState:
    """ODD 数学对象（2.1.3 类型冻结）：G_ODD(ODDState)≤0。
    诊断可用性与 OOD 分数不属于 X_phys；regime M_t 带时间戳（时态正确性）。"""
    x_phys: dict                       # {beta_N, q95, zeff, rad_frac, ...}
    regime: dict = field(default_factory=dict)   # multi-label 概率（HRM）
    t_regime: float = 0.0              # M_t 时间戳
    Q: float = 1.0
    d_avail: dict = field(default_factory=dict)  # {诊断: 0/1}
    ood: float = 0.0
    def regime_fresh(self, t_now):
        """(a) freshness：age(M_t)>age_max_M → regime 按未知处理（降档）。"""
        return (t_now - self.t_regime) <= constants.get("age_max_M", 0.1)
    def regime_consistency(self, fast_feats):
        """(b) consistency：快状态 regime 指示（mhd/ELM 标记）与 M_t 比对。
        fast_feats: {mhd_amp, elm_flag}；快状态已进新 regime 而 M_t 未更新 → False（以快状态为准）。"""
        if not self.regime:
            return True
        elm_fast = bool(fast_feats.get("elm_flag")) or fast_feats.get("mhd_amp", 0.0) > 0.05
        elm_m = self.regime.get("ELM", 0.0) > 0.5
        return elm_fast == elm_m

def g_odd(oddstate, t_now, fast_feats=None, shield=None):
    """G_ODD(ODDState)≤0 求值 → (ok, notes)。
    物理范围由 SafetyShield.ODD 判定；regime 不新鲜/不一致 → 保守判定（不允许特定 regime 授权）。"""
    notes = []
    sh = shield or SafetyShield()
    ok = sh.in_odd(oddstate.x_phys)
    if not ok:
        notes.append("X_phys 越出 ODD 物理范围")
    if not oddstate.regime_fresh(t_now):
        ok = False; notes.append(f"M_t stale(age>{constants.get('age_max_M', 0.1)}s)→regime 未知降档")
    if fast_feats is not None and not oddstate.regime_consistency(fast_feats):
        ok = False; notes.append("regime_consistency 失败：快状态已进入新 regime，M_t 未更新")
    for d, a in oddstate.d_avail.items():
        if not a:
            notes.append(f"诊断 {d} 不可用")
    if oddstate.ood > 0.99:
        notes.append(f"OOD 分数超阈 {oddstate.ood:.2f}")
    return ok and not notes, notes

# ================= 2.1.3：Safety Kernel（Safe Lookup + 局部线性化 CBF-QP） =================
class SafetyKernel:
    """Primary = Safe Lookup + 局部线性化 CBF-QP；Reachability = Offline Certification。
    离散时间 CBF：h(x_{t+1}) ≥ (1−γ)h(x_t)，h_j(x)=−g_j(x)−m_j（g_j=a_jᵀx−b_j 仿射）。
    QP 失败/超限/deadline → 纯查表保守动作（O(1) 快速路径）。"""
    def __init__(self, constraints, u_min, u_max, u_safe_table=None, gamma=None, maxiter=50):
        """constraints: [(a_j, b_j, m_j)]，a_j 对 x 的仿射向量、b_j 阈、m_j 概率裕量。"""
        self.cons = [(np.asarray(a, float), float(b), float(m)) for a, b, m in constraints]
        self.u_min, self.u_max = np.asarray(u_min, float), np.asarray(u_max, float)
        self.table = u_safe_table or {}
        self.gamma = constants.get("cbf_gamma", 1.0) if gamma is None else gamma
        self.maxiter = maxiter
    def h(self, x):
        """h(x)=min_j(−g_j(x)−m_j)（归一化坐标由调用方保证）。"""
        x = np.asarray(x, float)
        return min(-(a @ x - b) - m for a, b, m in self.cons)
    def safe_lookup(self, x):
        """O(1) 快速路径：离线安全动作表查表（默认零动作=不干预）。"""
        key = tuple(np.round(np.asarray(x, float), 1))
        return np.asarray(self.table.get(key, np.zeros_like(self.u_min)), float)
    def filter(self, u_cand, x, lin, m_lin=0.0, eps_lin=0.0, u_ref=None):
        """lin=(A,B,c)：x_{t+1}≈Ax+Bu+c（离线符号/autodiff 生成、在线查表，禁在线有限差分）。
        逐约束 discrete-time CBF：h_j(x_{t+1}) ≥ (1−γ)h_j(x_t)，h_j=−(a_j·x−b_j)−m_j。
        → a_j·B·u ≤ −a_j·(Ax+c)+b_j−m_j−(1−γ)h_j(x_t)（对 u 线性）。
        FEC-01：Taylor 余项显式进安全裕量——约束右端再减 m_lin（≥sup_{‖e‖≤εlin}|∇hᵀe|）；
        eps_lin > eps_lin_max（或工作点出局部有效域，由调用方以 eps_lin=inf 表达）→ 禁止 QP；
        Trust Region：动作盒收紧到 ‖u−u_ref‖≤tr_u（u_ref 默认 u_cand）。
        min ‖u−u_cand‖² s.t. 上述约束 + box；QP 失败/超限 → safe_lookup（O(1) 快速路径）。"""
        if eps_lin > constants.get("eps_lin_max", 0.05):
            return self.safe_lookup(x), "gate:lin"          # FEC-01 硬门：线性化不可信 → 禁 QP
        A, B, c = (np.asarray(v, float) for v in lin)
        x = np.asarray(x, float)
        if u_ref is None:                                   # 未给参考 → Trust Region 不收紧（向后兼容）
            lo, hi = self.u_min, self.u_max
        else:                                               # Trust Region ∩ 执行器盒
            u_ref = np.asarray(u_ref, float)
            tr = constants.get("tr_u", 0.5)
            lo = np.maximum(self.u_min, u_ref - tr)
            hi = np.minimum(self.u_max, u_ref + tr)
        u_cand = np.clip(np.asarray(u_cand, float), lo, hi)
        m_lin = np.broadcast_to(np.asarray(m_lin, float), (len(self.cons),))
        G, hsv = [], []
        for (a, b, m), ml in zip(self.cons, m_lin):
            hj_t = -(float(a @ x) - b) - m
            G.append(a @ B)
            hsv.append(float(-(a @ (A @ x + c)) + b - m - (1.0 - self.gamma) * hj_t - ml))
        G, hsv = np.array(G), np.array(hsv)
        from .safety_cpp import cbf_qp
        u, ok, backend = cbf_qp(u_cand, G, hsv, lo, hi, self.maxiter)
        if ok:
            return u, f"usafe_local:qp:{backend}"           # FEC-02：在线局部可行集（非完整认证集）
        return self.safe_lookup(x), "usafe_cert:lookup"     # FEC-02：离线认证安全动作集查表

# ---- FEC-03 SafetyConstraintSupport：约束状态依赖与 PhysicsOnly 覆盖矩阵 ----
CONSTRAINT_SUPPORT = {  # 类别 → (Required state, PhysicsOnly 支持, 支持条件, 不支持时动作)
    "fast":    ("X_fast", True,  "state valid + geometry valid", "继续 CBF-QP"),
    "mag":     ("磁状态 q95/β_N", True,  "EFIT/磁量 freshness 满足", "继续 CBF-QP"),
    "wall":    ("X_slow (W_wall/reservoir)", False, "需要有效 X_slow", "Safe PCS"),
    "transport": ("X_slow (D/V/nz 剖面)", False, "需要有效输运状态", "Safe PCS"),
}
def safety_support(categories, x_slow_valid, ident_ok=True):
    """激活约束类别集 → (SafetySupport, autonomous_ok)（2.2 四值枚举冻结）。
    FULL→允许；CONDITIONAL→须满足附加 Gate（wall/transport 的 gate=I^ident≠2，即 ident_ok）；PARTIAL/NONE→AutonomousControl=false。"""
    cats = list(categories)
    if any(CONSTRAINT_SUPPORT[c][1] is False and not x_slow_valid for c in cats):
        return "NONE", False
    if any(CONSTRAINT_SUPPORT[c][1] is False for c in cats):
        return ("CONDITIONAL", bool(ident_ok)) if x_slow_valid else ("PARTIAL", False)
    return "FULL", True

if __name__ == "__main__":
    s = SafetyShield()
    ok, dg, rs = s.check(dict(beta_N=1.5, q95=4.0, zeff=2.0, rad_frac=0.4), [], 0.5)
    assert ok and dg == 0
    ok2, dg2, rs2 = s.check(dict(beta_N=9.9), ["spec"], 0.5)
    assert not ok2 and dg2 >= 2 and any("AgeViolation" in r for r in rs2)
    for _ in range(6):
        s.check(dict(beta_N=1.5), [], 2.0)
    assert s.degrade >= 1   # 连续超阈降级
    s2 = SafetyShield()
    d = s2.check_decision(dict(beta_N=9.9), ["spec"], 0.5, state_age=0.03, ident=2)
    assert d.status == "rejected" and "ODD_VIOLATION" in d.reason_code
    assert "STALE_STATE" in d.reason_code and "IDENT_FAIL" in d.reason_code
    assert not d.eligibility_status and d.fallback_level >= 2
    d2 = SafetyShield().check_decision(dict(beta_N=1.5, q95=4.0, zeff=2.0, rad_frac=0.4), [], 0.5)
    assert d2.status == "approved" and d2.reason_code == ()
    E, Eu = eligibility({"Te": 0.001, "ne": 0.2}, {"Te": 0, "ne": 2}, {"Te": 0.9, "ne": 0.9})
    assert E["Te"] == 1 and E["ne"] == 0 and Eu == 0          # ne: ident=2 → 禁
    E2, Eu2 = eligibility({"Te": 0.001}, {"Te": 0}, {"Te": 0.9})
    assert Eu2 == 1
    # ---- 2.1.3 ODDState 时态门 ----
    od = ODDState(x_phys=dict(beta_N=1.5, q95=4.0, zeff=2.0, rad_frac=0.4),
                  regime={"ELM": 0.1, "H-mode": 0.9}, t_regime=10.0)
    ok0, n0 = g_odd(od, t_now=10.05, fast_feats=dict(mhd_amp=0.01))
    assert ok0 and not n0
    ok1, n1 = g_odd(od, t_now=10.5, fast_feats=dict(mhd_amp=0.01))     # age(M)=0.5>0.1
    assert not ok1 and any("stale" in s for s in n1)
    ok2, n2 = g_odd(od, t_now=10.05, fast_feats=dict(mhd_amp=0.2))      # 快状态 ELM，M_t 未更新
    assert not ok2 and any("consistency" in s for s in n2)
    # ---- 2.1.3 SafetyKernel ----
    # 单状态 x∈R、单动作 u∈R：x' = x + 0.5u；约束 g=x−1.0≤0（b=1.0），margin m=0.1
    kern = SafetyKernel(constraints=[(np.array([1.0]), 1.0, 0.1)], u_min=[-1.0], u_max=[1.0], gamma=1.0)
    A, B, c = np.array([[1.0]]), np.array([[0.5]]), np.array([0.0])
    u_ok, mode1 = kern.filter([0.0], [0.5], (A, B, c))                # h=0.4，0.5+0.5u≤0.9 → u≤0.8
    assert mode1.startswith("usafe_local:qp") and abs(u_ok[0]) <= 1.0
    u_bad, mode2 = kern.filter([1.0], [0.85], (A, B, c))              # h=0.05，0.85+0.5u≤0.9 → u≤0.1
    assert mode2.startswith("usafe_local:qp") and u_bad[0] <= 0.1 + 1e-6, u_bad
    u_fb, mode3 = kern.filter([1.0], [5.0], (A, B, c))                # 5+0.5u≤0.9 不可达 → 查表
    assert mode3 == "usafe_cert:lookup" and u_fb[0] == 0.0
    u_neg, mode4 = kern.filter([1.0], [0.95], (A, B, c), u_ref=[0.0])   # u_ref 在安全域：TR 盒 [−0.5,0.5]，u≤−0.1 拉回
    assert mode4.startswith("usafe_local:qp") and u_neg[0] <= -0.1 + 1e-6, u_neg
    # FEC-01：eps_lin 硬门 + m_lin 收紧 + Trust Region
    u_g, mode_g = kern.filter([0.0], [0.5], (A, B, c), eps_lin=0.2)     # eps_lin>eps_lin_max → 禁 QP
    assert mode_g == "gate:lin" and u_g[0] == 0.0
    u_m, mode_m = kern.filter([1.0], [0.85], (A, B, c), m_lin=0.05, u_ref=[0.0])  # m_lin 使界更紧
    u_m0, _ = kern.filter([1.0], [0.85], (A, B, c), u_ref=[0.0])
    assert u_m[0] <= u_m0[0] - 0.05 + 1e-6
    # FEC-03：SafetySupport 硬门
    from .safety import CONSTRAINT_SUPPORT, safety_support as _ss
    assert _ss(["fast", "mag"], x_slow_valid=False) == ("FULL", True)
    assert _ss(["wall"], x_slow_valid=False) == ("NONE", False)
    assert _ss(["wall"], x_slow_valid=True) == ("CONDITIONAL", True)   # 2.2：gate=ident_ok 默认 True
    assert _ss(["wall"], x_slow_valid=True, ident_ok=False) == ("CONDITIONAL", False)
    print("safety OK:", d.status, d.reason_code, "E_u", Eu, "| ODDState/Kernel OK")
