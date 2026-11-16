"""D/V 输运系数反演（MVP，ρ 坐标稳态输运，FEC-04 几何合同）。
正演：磁面平均稳态输运 (1/V′)∂(V′Γ)/∂ρ = S − n/τ，Γ = −D∂n/∂ρ + V·n（有限体积，三对角）。
反演：剖面测量 n_meas(ρ)（含噪）→ 参数网格搜索 θ=(D0,V0) 最小化剖面失配。
参数化（MVP）：D(ρ)=D0·(1+3ρ²)，V(ρ)=V0·ρ（V0<0 内向挤压）。"""
import numpy as np

def transport_solve(D, V, S, tau, rho, Vp, n_bc=1e8):
    """稳态输运正演 → n(ρ)。D,V,S,Vp 与 rho 同网格；tau 损失时间常数。
    有限体积：面通量 Γ=D(−n′)+V·n̄，体守恒 (1/V′)Δ(V′Γ)/Δρ = S − n/τ。"""
    N = len(rho)
    A = np.zeros((N, N)); b = S.copy()
    Df = 0.5 * (D[1:] + D[:-1]); Vf = 0.5 * (V[1:] + V[:-1])
    Vpf = 0.5 * (Vp[1:] + Vp[:-1])
    for i in range(1, N - 1):
        dρ = 0.5 * (rho[i + 1] - rho[i - 1])
        gp_D = -Df[i] / (rho[i + 1] - rho[i]); gp_V = 0.5 * Vf[i]       # Γ_{i+1/2} = gp_D·n_{i+1} + (−gp_D+gp_V)n_i? 见下展开
        gm_D = -Df[i - 1] / (rho[i] - rho[i - 1]); gm_V = 0.5 * Vf[i - 1]
        # Γ_{i+1/2} = gp_D·(n_{i+1}−n_i) + gp_V·(n_{i+1}+n_i)
        # Γ_{i−1/2} = gm_D·(n_i−n_{i−1}) + gm_V·(n_i+n_{i−1})
        A[i, i + 1] = Vpf[i] * (gp_D + gp_V) / dρ
        A[i, i] = (Vpf[i] * (-gp_D + gp_V) - Vpf[i - 1] * (gm_D + gm_V)) / dρ + Vp[i] / tau
        A[i, i - 1] = Vpf[i - 1] * (-gm_D + gm_V) / dρ
    # 芯部行：ρ→0 对称（Γ(0)=0 的有限体积首行）
    A[0, 0] = -Df[0] / (rho[1] - rho[0]) ** 2 - 0.5 * Vf[0] / (rho[1] - rho[0]) + Vp[0] / tau
    A[0, 1] = Df[0] / (rho[1] - rho[0]) ** 2 + 0.5 * Vf[0] / (rho[1] - rho[0])
    b[0] = S[0]
    A[-1, -1] = 1.0; A[-1, :-1] = 0.0; b[-1] = n_bc                    # 边缘 Dirichlet
    return np.linalg.solve(A, b)

def par_D(rho, D0): return D0 * (1.0 + 3.0 * rho ** 2)
def par_V(rho, V0): return V0 * rho

def invert(n_meas, rho, Vp, S, tau, grid_D0=None, grid_V0=None):
    """网格搜索反演 θ=(D0,V0) → (theta_hat, cost_surface, n_fit)。"""
    gD = np.linspace(0.3, 3.0, 28) if grid_D0 is None else grid_D0
    gV = np.linspace(-4.0, 0.0, 33) if grid_V0 is None else grid_V0
    C = np.full((len(gD), len(gV)), np.inf)
    for i, D0 in enumerate(gD):
        D = par_D(rho, D0)
        for j, V0 in enumerate(gV):
            try:
                n = transport_solve(D, par_V(rho, V0), S, tau, rho, Vp)
                if not np.all(np.isfinite(n)) or n.max() <= 0:
                    continue
                C[i, j] = float(np.mean((np.log(np.maximum(n, 1e-30)) -
                                         np.log(np.maximum(n_meas, 1e-30))) ** 2))
            except np.linalg.LinAlgError:
                continue
    i, j = np.unravel_index(np.argmin(C), C.shape)
    th = (gD[i], gV[j])
    n_fit = transport_solve(par_D(rho, th[0]), par_V(rho, th[1]), S, tau, rho, Vp)
    return th, C, n_fit

if __name__ == "__main__":
    rho = np.linspace(0.02, 1.0, 60)
    Vp = 4 * np.pi ** 2 * 1.7 * (0.6 * rho) ** 2 / 0.6 + 1.0   # 类环体积度规（量级即可）
    S = 1e18 * np.exp(-((rho - 0.92) / 0.05) ** 2)             # 边缘壳层源
    n_true = transport_solve(par_D(rho, 1.0), par_V(rho, -1.5), S, 1e-3, rho, Vp)
    rng = np.random.default_rng(3)
    n_meas = n_true * (1 + rng.normal(0, 0.02, len(rho)))
    th, C, n_fit = invert(n_meas, rho, Vp, S, 1e-3)
    err_D = abs(th[0] / 1.0 - 1); err_V = abs(th[1] / -1.5 - 1)
    assert err_D < 0.15 and err_V < 0.15, (th, err_D, err_V)
    print(f"dv_inversion OK: θ̂=(D0={th[0]:.2f},V0={th[1]:.2f}) vs θ*=(1.0,−1.5)，"
          f"误差 D {err_D*100:.1f}% / V {err_V*100:.1f}%")
