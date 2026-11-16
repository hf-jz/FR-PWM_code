"""训练标签制备：STRAHL 库 json → (帧, A 发射矩阵, n_z 标签, D/V 标签) npz。
A[line·pix? 不, A 为线级] 设计: I_line = Σ_i A[line,i]·n_z(ρ_i)，A[line,i]=fr_q(Te_i,ne_i)·PEC_i·chord_w_i。
帧合成与 L_spec 共用同一 A（同算子铁律：训练损失的正演 = 数据生成的正演）。
用法: /usr/bin/python3 -m frpwm.data.frame_labels <library.json> <out.npz>"""
import json, os, sys
import numpy as np

from frpwm.data.strahl_library import parse_pp
from frpwm.data.euv_frame import load_pec_tables, eq_fr
from frpwm.physics.adas import LINES

WL_GRID_VUV = np.linspace(200.0, 1700.0, 2048)    # VUV 碳波段（C III 977/1175, C IV 1549 等）
WL_GRID_EUV = np.linspace(40.0, 65.0, 2048)       # EUV 钨波段（W27+–W45+）
C_LINES = {k: v for k, v in LINES.items() if v[0] == "c"}
A_LCFS = 60.0


def build_case(r, n_tot, rho_bg, ne_bg, rho_te, te_bg, pecs, element="c",
               rho_out=None, n_s=96, r0_frac=0.0, c_frac=0.03, wl_grid=None):
    """单工况: 剖面插值到统一 ρ_out → A 矩阵 + 帧 + 标签。
    n_tot 幅度按 c_frac 物理定标（out_profiles.csv 为归一化形状——同 meas_ingest 的幅度假设纪律,
    形状来自 strahl 输运, 幅度=假设浓度, 标签显式记录）。"""
    if wl_grid is None:
        wl_grid = WL_GRID_VUV
    if rho_out is None:
        rho_out = np.linspace(0, 1.0, 32)
    a = A_LCFS
    nz_r = np.clip(r, 0, a) / a
    nz_on = np.interp(rho_out, nz_r, n_tot, left=n_tot[0], right=0.0)
    ne_on = np.interp(rho_out, rho_bg, ne_bg)
    core = rho_out <= 1.0
    scale = c_frac * ne_on[core].mean() / max(nz_on[core].mean(), 1e-300)
    nz_on = nz_on * scale
    # 弦几何（圆截面, 撞击 r0_frac·a）
    r0 = r0_frac * a
    L = np.sqrt(max(a * a - r0 * r0, 1.0))
    s = np.linspace(-L, L, n_s)
    rho_s = np.clip(np.sqrt(r0 * r0 + s * s) / a, 0.0, 1.05)
    te_s = np.interp(rho_s, rho_te, te_bg)
    ne_s = np.interp(rho_s, rho_bg, ne_bg)
    ds = abs(s[1] - s[0])
    # 弦权重: n_z(ρ_i) 对弦采样点的贡献（线性插值权重）
    W = np.zeros((len(rho_out), n_s))
    for i_s, rs in enumerate(rho_s):
        if rs >= rho_out[-1]:
            continue
        j = int(np.searchsorted(rho_out, rs) - 1)
        j = max(0, min(len(rho_out) - 2, j))
        f = (rs - rho_out[j]) / (rho_out[j + 1] - rho_out[j])
        W[j, i_s] += (1 - f) * ds
        W[j + 1, i_s] += f * ds
    fr_s = np.array([eq_fr(element, ne_s[i], te_s[i]) for i in range(n_s)])   # 每点一次, 全线复用
    lines_inten = {}
    Amat = np.zeros((len(pecs), len(rho_out)))
    wls = []
    for li, (label, (q, wl, tab)) in enumerate(pecs.items()):
        coef = np.zeros(n_s)          # 发射系数/单位 n_z
        for i in range(n_s):
            if q < fr_s.shape[1]:
                coef[i] = fr_s[i, q] * float(tab(ne_s[i], te_s[i]))
        Amat[li] = coef @ W.T
        wls.append(wl)
        lines_inten[wl] = float(Amat[li] @ nz_on)
    # 帧（高斯线型叠加, 与 euv_frame.frame 同公式；EUV 波段线窄 → 线宽随波段）
    sig = (0.15 if wl_grid[-1] < 200 else 0.35) / 2.355
    framev = np.zeros_like(wl_grid)
    for wl0, inten in lines_inten.items():
        framev += inten / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl_grid - wl0) / sig) ** 2)
    return dict(nz=nz_on, A=Amat, wls=np.array(wls), frame=framev, rho=rho_out)


def prep(library_json, out_npz, element="c", top_k=4, c_range=None):
    lib = json.load(open(library_json))
    rho_bg, ne_bg, rho_te, te_bg = parse_pp()
    if element == "w":
        from frpwm.physics.wlines import w_lines
        pecs = load_pec_tables(w_lines(top_k=top_k))
        wl_grid = WL_GRID_EUV
        c_lo, c_hi = (c_range or (-6.0, -4.0))          # W 典型 1e-6–1e-4
    else:
        pecs = load_pec_tables(C_LINES)
        wl_grid = WL_GRID_VUV
        c_lo, c_hi = (c_range or (-2.2, -1.1))          # C 典型 0.6%–8%
    cases = [c for c in lib["cases"] if c.get("ok")]
    rng = np.random.default_rng(lib.get("seed", 7) + 1)
    X, Yn, Yp, Aall = [], [], [], []
    for c in cases:
        c_frac = float(10 ** rng.uniform(c_lo, c_hi))
        d = build_case(np.array(c["r"]), np.array(c["n_tot"]),
                       rho_bg, ne_bg, rho_te, te_bg, pecs, element,
                       c_frac=c_frac, wl_grid=wl_grid)
        X.append(d["frame"]); Yn.append(d["nz"]); Aall.append(d["A"])
        Yp.append(list(c["params"]) + [c_frac])
    np.savez_compressed(out_npz, frames=np.array(X), nz=np.array(Yn),
                        params=np.array(Yp), A=np.array(Aall), rho=d["rho"],
                        wls=d["wls"], wl_grid=wl_grid)
    print(f"prep OK: {len(X)} 工况 → {out_npz} (帧{X[0].shape}, A{Aall[0].shape})")


def prep_t(library_json, out_npz, element="w", top_k=4, c_range=None):
    """时间分辨标签：库 json 带 n_tot_t/ts → 帧序列 (T,2048) + n_z(t,ρ) 标签。
    A 矩阵背景静态（Te/ne 不变）→ 逐时刻帧 = A @ n_z(t)（同算子铁律保持）。"""
    lib = json.load(open(library_json))
    rho_bg, ne_bg, rho_te, te_bg = parse_pp()
    from frpwm.physics.wlines import w_lines
    pecs = load_pec_tables(w_lines(top_k=top_k))
    wl_grid = WL_GRID_EUV
    c_lo, c_hi = (c_range or (-6.0, -4.0))
    cases = [c for c in lib["cases"] if c.get("ok") and "n_tot_t" in c]
    rng = np.random.default_rng(lib.get("seed", 7) + 1)
    X, Yn, Yp, Aall, T = [], [], [], [], None
    for c in cases:
        c_frac = float(10 ** rng.uniform(c_lo, c_hi))
        n_t = np.array(c["n_tot_t"])                       # (T, R)
        d_last = build_case(np.array(c["r"]), n_t[-1],
                            rho_bg, ne_bg, rho_te, te_bg, pecs, element,
                            c_frac=c_frac, wl_grid=wl_grid)
        # 逐时刻: 幅度定标沿用末时刻 scale（同一工况同一浓度）
        r_arr = np.array(c["r"]); a = A_LCFS
        nz_r = np.clip(r_arr, 0, a) / a
        rho_out = d_last["rho"]
        ne_on = np.interp(rho_out, rho_bg, ne_bg)
        core = rho_out <= 1.0
        nz_t = np.array([np.interp(rho_out, nz_r, row, left=row[0], right=0.0) for row in n_t])
        scale = c_frac * ne_on[core].mean() / max(nz_t[-1][core].mean(), 1e-300)
        nz_t = nz_t * scale                                # (T, 32)
        frames_t = np.einsum("lk,tk->tl", d_last["A"], nz_t)   # (T, 65) 线强
        sig = 0.15 / 2.355
        spec_t = np.zeros((len(nz_t), len(wl_grid)))
        for li, wl0 in enumerate(d_last["wls"]):
            spec_t += frames_t[:, li:li + 1] / (sig * np.sqrt(2 * np.pi)) * \
                np.exp(-0.5 * ((wl_grid - wl0) / sig) ** 2)
        X.append(spec_t); Yn.append(nz_t); Aall.append(d_last["A"])
        Yp.append(list(c["params"]) + [c_frac])
        T = np.array(c["ts"])
    np.savez_compressed(out_npz, frames=np.array(X), nz=np.array(Yn),
                        params=np.array(Yp), A=np.array(Aall), rho=d_last["rho"],
                        wls=d_last["wls"], wl_grid=wl_grid, ts=T)
    print(f"prep_t OK: {len(X)} 工况 × T={len(T)} → {out_npz} (帧{X[0].shape}, A{Aall[0].shape})")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "t":
        prep_t(sys.argv[1], sys.argv[2])
    else:
        prep(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "c")
