"""非平衡碰撞辐射(CR)求解器：dn_q/dt = ne·[S_{q-1}n_{q-1} − (S_q+α_q)n_q + α_{q+1}n_{q+1}]
速率系数 = ADF11 scd/acd（strahl newdat 本地，C/W 全）；隐式三对角时间推进（刚性稳定）。
用途：L1 快演化动力学标签生成 + L_CR 物理一致性残差（PINN 训练损失）。
自检：t→∞ 应收敛到 adas.ion_fractions 电离平衡。"""
import math, os, sys
from .._env import external  # noqa: E402
import numpy as np
from scipy.linalg import solve_banded

_NEWDAT = external("newdat", "STRAHL_ROOT", "path of the STRAHL source tree")
_SPEC2CONC = external("", "SPEC2CONC", "path of the plasma_spec2conc package")
if _SPEC2CONC not in sys.path:
    sys.path.insert(0, _SPEC2CONC)
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
import adas_pec  # noqa: E402

_TAB = {}


def _tables(element):
    """(log10 ne 网格, log10 Te 网格, S[Z], α[Z]) — 值线性化, 缓存幂等。"""
    if element in _TAB:
        return _TAB[element]
    scd = adas_pec.parse_adf11(os.path.join(_NEWDAT, adas_pec.ELEM_SCD[element]))
    acd = adas_pec.parse_adf11(os.path.join(_NEWDAT, adas_pec.ELEM_ACD[element]))
    ne_g, te_g, S = scd
    _, _, A = acd
    _TAB[element] = (np.array(ne_g), np.array(te_g), np.array(S), np.array(A))
    return _TAB[element]


def _interp2(ne_g, te_g, m, lne, lte):
    """log10 双线性（网格对齐同 96 系列）；m[nt][nd]。"""
    i = int(np.searchsorted(ne_g, lne) - 1)
    j = int(np.searchsorted(te_g, lte) - 1)
    i = max(0, min(len(ne_g) - 2, i)); j = max(0, min(len(te_g) - 2, j))
    fx = np.clip((lne - ne_g[i]) / max(ne_g[i + 1] - ne_g[i], 1e-30), 0, 1)
    fy = np.clip((lte - te_g[j]) / max(te_g[j + 1] - te_g[j], 1e-30), 0, 1)
    a = m[j][i] * (1 - fx) + m[j][i + 1] * fx
    b = m[j + 1][i] * (1 - fx) + m[j + 1][i + 1] * fx
    return 10.0 ** (a * (1 - fy) + b * fy)   # ADF11 值为 log10 → 线性化（log 域插值后 pow10 保正性）


def rates(element, ne_cm3, te_eV):
    """S_q (q=0..Z-1 电离), α_q (q=1..Z 复合) [cm³/s] → (S[Z+1], A[Z+1]) 对齐到价态索引。"""
    ne_g, te_g, S, A = _tables(element)
    lne, lte = math.log10(ne_cm3), math.log10(te_eV)
    z = len(S)
    Sv = np.zeros(z + 1); Av = np.zeros(z + 1)
    for q in range(z):
        Sv[q] = _interp2(ne_g, te_g, S[q], lne, lte)
    for q in range(1, z + 1):
        Av[q] = _interp2(ne_g, te_g, A[q - 1], lne, lte)
    return Sv, Av


def cr_step(n, dt, ne_cm3, te_eV, element):
    """隐式欧拉一步（三对角）：(I − dt·ne·M) n' = n。n 归一化分布。"""
    S, A = rates(element, ne_cm3, te_eV)
    z1 = len(n)
    lo = np.zeros(z1); di = np.ones(z1); up = np.zeros(z1)
    for q in range(z1):
        gain_ion = ne_cm3 * S[q - 1] if q >= 1 else 0.0     # 从 q-1 电离来
        loss = ne_cm3 * (S[q] + A[q])                        # 电离走+复合走
        gain_rec = ne_cm3 * A[q + 1] if q + 1 < z1 else 0.0  # 从 q+1 复合来
        di[q] = 1.0 + dt * loss
        if q >= 1:
            lo[q] = -dt * gain_ion
        if q + 1 < z1:
            up[q] = -dt * gain_rec
    ab = np.zeros((3, z1)); ab[0, 1:] = up[:-1]; ab[1] = di; ab[2, :-1] = lo[1:]
    n2 = solve_banded((1, 1), ab, n)
    n2 = np.maximum(n2, 0.0)
    return n2 / max(n2.sum(), 1e-300)


def cr_solve(element, ne_cm3, te_eV, t_end, dt=None, n0=None):
    """时间演化 → (t 数组, n(t, q))；n0 缺省全中性。"""
    z1 = len(_tables(element)[2]) + 1
    n = np.zeros(z1) if n0 is None else np.asarray(n0, float)
    if n0 is None:
        n[0] = 1.0
    n = n / n.sum()
    # 自适应步长：max 速率倒数 / 20（刚性安全），上限 t_end/200
    S, A = rates(element, ne_cm3, te_eV)
    tau = 1.0 / max(ne_cm3 * max(S.max(), A.max(), 1e-300), 1e-300)
    dt = dt or min(tau / 20.0, t_end / 200.0)
    steps = max(2, int(math.ceil(t_end / dt)))
    dt = t_end / steps
    ts = np.linspace(0, t_end, steps + 1)
    out = np.zeros((steps + 1, z1)); out[0] = n
    for k in range(steps):
        n = cr_step(n, dt, ne_cm3, te_eV, element)
        out[k + 1] = n
    return ts, out


def cr_residual(n_prev, n_next, dt, ne_cm3, te_eV, element):
    """L_CR 物理一致性残差（PINN 损失用）：‖n_next − cr_step(n_prev)‖²。"""
    pred = cr_step(np.asarray(n_prev, float), dt, ne_cm3, te_eV, element)
    d = np.asarray(n_next, float) - pred
    return float(np.mean(d * d))


if __name__ == "__main__":
    from frpwm.physics.adas import ion_fractions
    for el, q_check, t_end, dt in (("c", 3, 5e-3, None), ("w", 40, 0.5, 2.5e-3)):
        ts, out = cr_solve(el, 1e13, 2000.0, t_end, dt=dt)
        eq = dict(ion_fractions(el, 1e13, 2000.0))
        n_end = out[-1]
        # 与平衡比较主态
        q_dom_cr = int(np.argmax(n_end))
        q_dom_eq = max(eq, key=eq.get)
        print(f"{el}: CR 主态 q={q_dom_cr} vs 平衡主态 q={q_dom_eq}, "
              f"末端分布Σ={n_end.sum():.4f}, 步数={len(ts)-1}")
        assert q_dom_cr == q_dom_eq, f"{el} CR 未收敛到平衡主态"
        r = cr_residual(out[-2], out[-1], ts[-1] - ts[-2], 1e13, 2000.0, el)
        print(f"  L_CR 自洽残差(末步)={r:.3e}")
    print("cr OK")
