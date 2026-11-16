"""FR-PWM 2.1 数据 Schema v1（附录 B）：provenance 十元组 + 核心状态量。
所有跨模块数据必须用 Quantity 携带 provenance，禁止裸数值。"""
import time
from dataclasses import dataclass, field, replace
import numpy as np

# X^phys 分量清单（main.pdf 类型映射 X^phys=T(x*)={X_fast,X_slow} 冻结，plan_eval09141034）：
# 不得出现既不在 X_fast 也不在 X_slow 的第三个物理状态分量
X_FAST_COMPONENTS = ("T_e", "n_e", "T_i", "Z_eff", "P_rad", "MHD", "beta_N", "q", "n_z^q", "X_HDT")
X_SLOW_COMPONENTS = ("D", "V", "S", "W_wall")
# n_z^q 层级归属冻结（2.2 物理依据）：电离/复合是快过程（τ_ion ≪ τ_transport），
# 电荷态分布属 X^fast；慢环只辨识输运系数 + 壁库存（并对同一 n_z^q 给 refined revision，
# 经 runtime.prediction_cache.nz_q_authoritative 合并，不新增状态分量、不产生第二个真值源）。

@dataclass
class Provenance:
    t_measure: float
    t_available: float
    source: str
    quality: float = 1.0
    uncertainty: float = 0.0
    imputation: bool = False
    estimation_mode: str = "measured"            # 2.1.3 FVP：measured/filtered/predicted/propagated/imputed/fallback
    schema_version: int = 4                     # u32，Schema v1.4
    artifact_version: str = "frpwm-2.2"         # 工件语义版本
    bundle_id: str = ""                         # bytes32 SHA-256（空=未签）
    revision: int = 0                           # u64 状态快照修订号
    EST_MODES = ("measured", "filtered", "predicted", "propagated", "imputed", "fallback")
    def __post_init__(self):
        assert self.estimation_mode in self.EST_MODES, self.estimation_mode
        # fallback/imputed 强制降档（与 quality 降权等价的显式语义层，附录 B 冻结）
        if self.estimation_mode in ("imputed", "fallback"):
            self.quality = min(self.quality, 0.3)
    @property
    def version(self):                          # 兼容别名（旧调用点）
        return self.artifact_version
    @property
    def latency(self): return self.t_available - self.t_measure
    def age(self, t_now=None): return (t_now or time.time()) - self.t_measure

@dataclass
class Quantity:
    """任一物理量 = value + provenance。imputed=True 自动降信任。"""
    value: np.ndarray
    prov: Provenance
    def __post_init__(self):
        self.value = np.asarray(self.value, dtype=np.float32)
        if self.prov.imputation:
            self.prov.quality = min(self.prov.quality, 0.5)
            self.prov.uncertainty = max(self.prov.uncertainty, 0.15)
    def stale(self, source_key, registry_age_max, t_now=None):
        from . import constants
        amax = registry_age_max.get(source_key, 0.05)
        return self.prov.age(t_now) > amax
    def with_value(self, v, **kw):
        return Quantity(np.asarray(v, dtype=np.float32), replace(self.prov, **kw))

def now_prov(source, uncertainty=0.0, quality=1.0, imputation=False, t=None):
    t = t or time.time()
    return Provenance(t_measure=t, t_available=t, source=source,
                      quality=quality, uncertainty=uncertainty, imputation=imputation)

@dataclass
class XFast:   # 快物理状态（ρ 网格剖面 + 全局量）
    rho: np.ndarray
    Te: Quantity; ne: Quantity; Ti: Quantity
    Zeff_los: Quantity; P_rad: Quantity
    beta_N: Quantity; q95: Quantity
    w_neural: np.ndarray = None   # 来源权重（逐量记录）
    w_observer: np.ndarray = None

@dataclass
class XSlow:   # 慢物理参数（TRM/反演给出）
    rho: np.ndarray
    nz_prof: object = None    # {species: {q: Quantity(profile)}} 各电离态密度
    N_tot: object = None      # {species: 总含量}
    C_avg: object = None      # {species: 平均浓度}
    W_wall: float = 0.0
    ident: int = 0            # 0 可辨识 / 1 弱 / 2 不可辨识

@dataclass
class RiskEvent:  # 事件级风险 {P, TTE, severity, uncertainty}
    P: float; tte: float; severity: float; unc: float; name: str = ""

@dataclass
class MetaState:
    sigma: dict = field(default_factory=dict)   # 五源
    Q: float = 1.0                              # 唯一状态质量分
    risks: list = field(default_factory=list)   # [RiskEvent]
    regime: dict = field(default_factory=dict)  # multi-label 概率
    degrade: int = 0                            # L0..L4
