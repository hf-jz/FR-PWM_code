"""输运不变量数值容差表（2.1.3，附录 A Physics Invariant Tolerance 的代码形式）。
三级处置：|r|<ε 数值合法放行；ε≤|r|<10ε 记录 warning 计入 Q；|r|≥10ε 物理非法回滚。"""
import numpy as np
from .. import constants

TOL = constants.get("inv_tol")

def r_charge(nz_q):
    """电荷态平衡：|Σ_q n_z^q − n_z| / n_z。"""
    nz_q = np.asarray(nz_q, float)
    tot = nz_q.sum(0)
    nz = np.maximum(nz_q.sum(0), 1e-30)
    return float(np.abs(tot - nz).max() / nz.max())

def r_particle(dN, flux_in, flux_out, s_wall, s_gas, dt, N):
    """粒子守恒：|ΔN − (Γ_in−Γ_out+S_wall+S_gas)Δt| / N。"""
    return float(abs(dN - (flux_in - flux_out + s_wall + s_gas) * dt) / max(abs(N), 1e-30))

def r_boundary(flux_plus, flux_minus):
    """边界通量闭合：|Γ⁺−Γ⁻| / |Γ|。"""
    return float(abs(flux_plus - flux_minus) / max(abs(flux_plus), abs(flux_minus), 1e-30))

def r_source(int_S, s_wall, s_gas, s_lbo, s_rec):
    """源项库存收支：|∫S_z dV − Σ 分项| / |∫S_z dV|。"""
    return float(abs(int_S - (s_wall + s_gas + s_lbo + s_rec)) / max(abs(int_S), 1e-30))

def verdict(r, kind):
    """三级处置 → 'ok' / 'warning' / 'rollback'。kind ∈ TOL 键。"""
    eps = TOL[kind]
    if r < eps:
        return "ok"
    if r < 10 * eps:
        return "warning"
    return "rollback"

def check_all(residuals):
    """residuals: {kind: r} → (worst_action, {kind: action})。worst=rollback 优先。"""
    acts = {k: verdict(v, k) for k, v in residuals.items()}
    worst = "rollback" if "rollback" in acts.values() else (
        "warning" if "warning" in acts.values() else "ok")
    return worst, acts

if __name__ == "__main__":
    nz_q = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert r_charge(nz_q) < TOL["charge"] and verdict(r_charge(nz_q), "charge") == "ok"
    assert verdict(5e-6, "charge") == "warning" and verdict(5e-5, "charge") == "rollback"
    rp = r_particle(dN=100.0, flux_in=1e5, flux_out=0.0, s_wall=0.0, s_gas=0.0, dt=1e-3, N=1e6)
    assert verdict(rp, "particle") == "ok"
    rb = r_boundary(1.001, 1.0)
    assert verdict(rb, "boundary") == "warning"
    rs = r_source(100.0, 50.0, 30.0, 10.0, 10.5)
    assert verdict(rs, "source") in ("ok", "warning")
    worst, acts = check_all({"charge": 0.0, "particle": 0.05, "boundary": 0.0})
    assert worst == "rollback" and acts["particle"] == "rollback"
    print(f"invariants OK: 容差表 {TOL}，三级处置/worst 聚合全过")
