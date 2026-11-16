"""壁状态记忆模型（P1-9，报告 16 章：dW/dt = Γ_dep − W/τ_wall − Γ_desorb，Γ_fuel=R_w·Γ_desorb）。
Markov 近似的记忆慢变量，支撑杂质积累预测。"""
import numpy as np

class WallReservoir:
    def __init__(self, tau_wall=2.0, R_w=0.9, W0=0.0):
        self.tau, self.Rw, self.W = tau_wall, R_w, W0
        self.hist = []
    def step(self, gamma_dep, gamma_desorb, dt):
        """ODE 前向欧拉（dt s）。Γ_dep 沉积入壁，W/τ 滞留释放，Γ_desorb 解吸。"""
        dW = gamma_dep - self.W / self.tau - gamma_desorb
        self.W = max(self.W + dW * dt, 0.0)
        gamma_fuel = self.Rw * gamma_desorb
        self.hist.append((self.W, gamma_fuel))
        return self.W, gamma_fuel
    def steady(self, gamma_dep, gamma_desorb):
        return self.tau * max(gamma_dep - gamma_desorb, 0.0)

if __name__ == "__main__":
    w = WallReservoir(tau_wall=1.0, R_w=0.9)
    for _ in range(2000):
        W, gf = w.step(1.0, 0.2, 0.005)               # 恒定沉积 → 收敛到稳态
    assert abs(W - w.steady(1.0, 0.2) - 0.0) / max(w.steady(1.0, 0.2), 1e-9) < 0.05, (W, w.steady(1.0, 0.2))
    assert W >= 0 and abs(gf - 0.18) < 1e-9
    print(f"wall OK: W_steady={W:.3f} (理论 {w.steady(1.0,0.2):.3f}), Γ_fuel={gf:.3f}")
