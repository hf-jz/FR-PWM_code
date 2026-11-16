"""全帧光谱合成器（EUV/VUV）：(ρ 网格 Te/ne/n_z 剖面) × 谱线表 → 弦积分线强 → 2048px 帧。
与逐线模式的差别：一次产出整帧 I(λ)（SpectrumSynth.reconstruct 高斯线型叠加+连续谱本底）。
PEC 直读对应价态 ADF15 文件（绕开 PecCache 全目录扫描，批量库场景快 ~40×）。"""
import os, sys
from .._env import external  # noqa: E402
import numpy as np

_SPEC2CONC = external("", "SPEC2CONC", "path of the plasma_spec2conc package")
if _SPEC2CONC not in sys.path:
    sys.path.insert(0, _SPEC2CONC)
import adas_pec  # noqa: E402

from frpwm.physics.adas import _adf15_dir
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.physics.cr import rates


def eq_fr(element, ne_cm3, te_eV):
    """平衡电离态分布（速率比递推 n_{q+1}/n_q = S_q/α_{q+1}，与 CR 求解器同一速率表→同算子铁律）。"""
    S, A = rates(element, ne_cm3, te_eV)
    z = len(S) - 1                       # rates() 已按价态索引 padded (S[Z]=0, A[0]=0)
    n = np.zeros(z + 1); n[0] = 1.0
    for q in range(z):
        n[q + 1] = n[q] * S[q] / max(A[q + 1], 1e-300)
    return n / n.sum()


def load_pec_tables(lines):
    """{label: (q, wl, PecTable)}；lines = {label: (element, q, wl)}（wlines.w_lines() 或 adas.LINES 子集）。
    文件优选 pju(EXCIT)>pjr>llu>llr（与 adas._find_file 同序），tol=0.6。"""
    out = {}
    for label, (el, q, wl) in lines.items():
        d = _adf15_dir(el)
        cands = sorted(os.listdir(d))
        f = None
        for pref in ("_pju#", "_pjr#", "_llu#", "_llr#", "_ca#", ""):
            hit = [fn for fn in cands if pref in fn and fn.endswith(f"#{el}{q}.dat")]
            if hit:
                f = os.path.join(d, hit[0])
                break
        if f is None:
            continue
        try:
            t = adas_pec.find_pec(f, wl, tol=0.6) if f else None
        except Exception:
            t = None
        if t is None:                       # 优选文件无此线 → 全目录扫描（一次性成本, 与 PecCache 同路径）
            from frpwm.physics.adas import _find_file
            try:
                t = _find_file(el, wl, q)
            except Exception:
                t = None
        if t is not None:
            out[label] = (q, wl, t)
    return out


def frame(rho_a, te_prof, ne_prof, nz_prof, pecs, element,
          r0_frac=0.0, wl_grid=None, fwhm_ang=0.15, cont_level=0.0, n_s=96):
    """圆截面弦（撞击参数 r0_frac·a）弦积分 → I(λ) 帧。
    rho_a=(rho 网格 0..1+, a_cm)；te/ne/nz 在 rho 网格上。返回 (wl_grid, I)。"""
    rho, a_cm = rho_a
    if wl_grid is None:
        wl_grid = np.linspace(40.0, 65.0, 2048)
    r0 = r0_frac * a_cm
    L = np.sqrt(max(a_cm * a_cm - r0 * r0, 1.0))
    s = np.linspace(-L, L, n_s)
    r_s = np.sqrt(r0 * r0 + s * s)
    rho_s = np.clip(r_s / a_cm, 0.0, rho.max())
    te_s = np.interp(rho_s, rho, te_prof)
    ne_s = np.interp(rho_s, rho, ne_prof)
    nz_s = np.interp(rho_s, rho, nz_prof)
    ds_cm = abs(s[1] - s[0])
    intens = {}
    for label, (q, wl, tab) in pecs.items():
        eps = np.zeros(n_s)
        for i in range(n_s):
            fr = eq_fr(element, ne_s[i], te_s[i])
            if q < len(fr):
                pec = float(tab(ne_s[i], te_s[i]))
                eps[i] = nz_s[i] * fr[q] * pec
        intens[wl] = float(np.trapz(eps, dx=ds_cm))
    sig = fwhm_ang / 2.355
    I = np.full_like(wl_grid, cont_level, dtype=float)
    for wl0, inten in intens.items():
        I += inten / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl_grid - wl0) / sig) ** 2)
    return wl_grid, I


if __name__ == "__main__":
    from frpwm.physics.wlines import w_lines
    rho = np.linspace(0, 1.1, 57)
    te = 2000.0 * (1 - np.clip(rho, 0, 1) ** 2) + 50
    ne = 8e13 * (1 - 0.6 * np.clip(rho, 0, 1) ** 2) + 1e12
    nz = 1e11 * np.exp(-((np.clip(rho, 0, 1) - 0.5) / 0.2) ** 2)
    pecs = load_pec_tables(w_lines(top_k=2))
    print("PEC 表:", len(pecs))
    wl, I = frame((rho, 60.0), te, ne, nz, pecs, "w")
    assert I.max() > 0 and len(wl) == 2048
    print(f"frame OK: 峰值 {I.max():.3e} @ {wl[np.argmax(I)]:.2f}Å, 非零线 {sum(1 for v in [I] if True)}")
