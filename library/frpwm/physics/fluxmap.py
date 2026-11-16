"""磁面 map（流水线步 4）：(R,Z)↔ρ 双向映射，剖面量沿弦取值。"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class FluxMap:
    def __init__(self, g):
        self.g = g
        psi_n = (g["PSIRZ"] - g["PSIMAG"]) / (g["PSIBCY"] - g["PSIMAG"] + 1e-30)
        self.rho2 = np.clip(psi_n, 0.0, None)          # ρ² 网格
        self._itp = RegularGridInterpolator(
            (g["ZGRID"], g["RGRID"]), self.rho2, bounds_error=False, fill_value=4.0)
    def rho(self, R, Z):
        """(R,Z)→ρ（LCFS 外 ρ>1，外推区给 ρ=2 截断）。"""
        return np.sqrt(self._itp(np.c_[np.atleast_1d(Z), np.atleast_1d(R)]).clip(0, 4.0))
    def inside(self, R, Z):
        return self.rho(R, Z) <= 1.0

if __name__ == "__main__":
    from .gfile import generate_gfile
    fm = FluxMap(generate_gfile(R0=1.7, a=0.6))
    assert abs(fm.rho(1.7, 0.0)[0]) < 1e-3           # 磁轴 ρ≈0
    assert abs(fm.rho(2.3, 0.0)[0] - 1.0) < 0.05     # 中平面 LCFS ρ≈1
    assert fm.rho(3.0, 0.0)[0] > 1.0                 #  scrape-off
    print("fluxmap OK")
