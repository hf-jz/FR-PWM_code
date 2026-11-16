"""杂质浓度体积分（流水线步 12）：N_z=∫n_z dV，C_z=∫n_z dV/∫n_e dV。"""
import numpy as np

def volume_elements(g, rho):
    """圆环近似体积元：dV = 2πR · 2πr dr（环向周长 × 极向环带），按 ρ 网格。
    圆截面: r(ρ)=a·ρ，R(ρ)≈R0（一阶）。返回各网格点体积元 [m³]。"""
    R0 = float(g["RMAXIS"]); a = float(np.max(g["RBBBS"]) - g["RMAXIS"])
    r = a * rho
    dr = np.gradient(r) if len(r) > 1 else np.array([a])
    dv = (2 * np.pi * R0) * (2 * np.pi * r) * dr
    dv[0] = (2 * np.pi * R0) * np.pi * (r[1] if len(r) > 1 else a) ** 2 * 0.5  # 芯盘
    return dv

def concentration(nz_prof_cm3, ne_prof_cm3, g, rho):
    """n_z(ρ), n_e(ρ) [cm⁻³] → (N_z [无量纲-粒子数], C_z [平均浓度比])。"""
    dv = volume_elements(g, rho) * 1e6          # m³ → cm³
    Nz = float(np.sum(nz_prof_cm3 * dv))
    Ne = float(np.sum(ne_prof_cm3 * dv))
    return Nz, Nz / max(Ne, 1e-30)

if __name__ == "__main__":
    from .gfile import generate_gfile
    rho = np.linspace(0, 1, 30)
    g = generate_gfile(R0=1.7, a=0.6)
    Nz, Cz = concentration(np.full(30, 1e16), np.full(30, 5e19), g, rho)
    assert 0 < Cz < 0.01 and Nz > 0
    # 均匀剖面时 C_z 应严格等于密度比
    assert abs(Cz - 1e16 / 5e19) / (1e16 / 5e19) < 0.1, Cz
    print("concentration OK:", Cz)
