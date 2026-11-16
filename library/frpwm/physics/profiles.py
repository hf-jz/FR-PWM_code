"""分布参数插值（流水线步 3）：剖面统一插值到 ρ∈[0,1] 网格，保形防负值。"""
import numpy as np

N_RHO = 120   # 网格须分辨边缘发射壳层（C III 977 在 Te~50eV、ρ≈1 处 peaked），粗网格会 4× 失真

def rho_grid(n=N_RHO):
    return np.linspace(0, 1, n)

def interp_profile(rho_src, val_src, rho_dst, nonnegative=True):
    """保形插值：线性 + 非负截断（密度/温度物理非负）。"""
    v = np.interp(rho_dst, rho_src, val_src)
    return np.clip(v, 0, None) if nonnegative else v

def typical_profiles(rho):
    """典型 L-mode 形状剖面（合成数据/测试用）：Te 3keV 芯、ne 5e19 芯、Ti≈0.8Te。"""
    Te = 3000.0 * (1 - rho**2) ** 0.7 + 50.0
    ne = 5.0e19 * (1 - rho**2) ** 0.3 + 2.0e18
    Ti = 0.8 * Te
    return Te, ne, Ti

if __name__ == "__main__":
    r = rho_grid()
    Te, ne, Ti = typical_profiles(r)
    Te2 = interp_profile(r, Te, rho_grid(50))
    assert (Te2 > 0).all() and Te2[0] > Te2[-1]
    print("profiles OK")
