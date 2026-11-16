"""FEC-04 D/V 坐标合同：r 坐标与 ρ 坐标的工程接口对象。
TransportGeometry{psi_axis, psi_LCFS, rho_grid, Vprime_rho, dr_drho, geometry_id}；
TransportParams{D_phys, V_phys, source, loss, units}——两对象均进 provenance/BundleID。
硬规则：禁止同名 solve_transport_r()/solve_transport_rho() 共享同一 D,V 而无显式 conversion object。"""
import hashlib
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class TransportGeometry:
    psi_axis: float
    psi_LCFS: float
    rho_grid: tuple            # ρ 网格（含 SOL 加密）
    Vprime_rho: tuple          # V'(ρ)=dV_plasma/dρ [m³]
    dr_drho: tuple             # dr/dρ 几何因子（∂/∂r = (dρ/dr)∂/∂ρ 的逆）
    geometry_id: str = ""
    def __post_init__(self):
        if not self.geometry_id:
            h = hashlib.sha256(repr((round(self.psi_axis, 9), round(self.psi_LCFS, 9),
                                     len(self.rho_grid))).encode()).hexdigest()[:12]
            object.__setattr__(self, "geometry_id", f"geo-{h}")
    def psi_N(self, psi):
        """轴心基准归一化磁通（2.1.3 FVP 冻结）：ψ_N=(ψ−ψ_axis)/(ψ_LCFS−ψ_axis)。"""
        return (np.asarray(psi) - self.psi_axis) / (self.psi_LCFS - self.psi_axis + 1e-30)
    def dDr(self, dD_drho, rho):
        """梯度变换：∂/∂r = (dρ/dr)∂/∂ρ → 乘 dr/dρ 的逆（逐点）。"""
        dr = np.interp(rho, self.rho_grid, self.dr_drho)
        return np.asarray(dD_drho) / np.maximum(dr, 1e-30)

@dataclass(frozen=True)
class TransportParams:
    D_phys: tuple              # SI m²/s（r 或 ρ 网格，配合 geometry 解释）
    V_phys: tuple              # SI m/s
    source: tuple
    loss: tuple
    units: str = "SI"          # 物理系数保留 SI 含义，禁止无量纲网络输出直接当 D/V

def from_gfile(g, rho):
    """由 gfile 构造 TransportGeometry（V'(ρ) 由体积元差分，dr/dρ 由中平面几何）。"""
    from .concentration import volume_elements
    dv = volume_elements(g, rho)                       # 各 ρ 点体积元 [m³]（与 rho 同长）
    V = np.cumsum(dv)
    Vp = np.gradient(V, rho, edge_order=1)             # V'(ρ)=dV/dρ
    R0 = float(np.mean(g["RBBBS"])); a = float((g["RBBBS"].max() - g["RBBBS"].min()) / 2)
    dr = np.full_like(rho, a)                          # 中平面 r≈R0+aρ → dr/dρ=a（一阶几何）
    return TransportGeometry(psi_axis=float(g["PSIMAG"]), psi_LCFS=float(g["PSIBCY"]),
                             rho_grid=tuple(float(x) for x in rho),
                             Vprime_rho=tuple(float(x) for x in Vp),
                             dr_drho=tuple(float(x) for x in dr))

if __name__ == "__main__":
    from .gfile import generate_gfile
    from .profiles import rho_grid
    rho = rho_grid()
    tg = from_gfile(generate_gfile(R0=0.85, a=0.55), rho)
    assert tg.geometry_id.startswith("geo-") and len(tg.Vprime_rho) == len(rho)
    assert np.all(np.array(tg.Vprime_rho) > 0)         # 体积随 ρ 单调增
    assert abs(tg.psi_N(tg.psi_axis)) < 1e-12 and abs(tg.psi_N(tg.psi_LCFS) - 1.0) < 1e-12
    tp = TransportParams(D_phys=(1.0,), V_phys=(0.0,), source=(0.0,), loss=(0.0,))
    assert tp.units == "SI"
    print(f"geometry OK: {tg.geometry_id} V'(ρ)>0, ψ_N 锚点精确")
