"""T2-B 四种观测组合的可辨识性：单/多价态 × 稳态/时间分辨 → Fisher 谱与 CRLB。

审稿意见 Stage 1 T2-B/T2-C 与 Stage 3 §七 要求：不是断言"时间分辨打破稳态简并"，
而是量化"在何种观测组合下，时间分辨真正提高了 (D, V) 的可辨识性"。
观测算子 = 与训练同一 A 矩阵（line·ρ）：I_l(t) = A_l · n_z(t)；
同一工况族参数 (Dmax, v_outer)，噪声 3% 乘性独立 → Σ = diag((0.03 I)²)；
参数按物理尺度归一 (Dmax/4, v/15) → Fisher 无量纲，σ 即相对精度。
输出: tests/results/ident4/ident4_report.json
用法: /usr/bin/python3 tests/ident_4combos.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frpwm.physics.transport import run_forward_dv
from frpwm.data.strahl_library import parse_pp
from frpwm.data.frame_labels import build_case, C_LINES, WL_GRID_VUV, A_LCFS
from frpwm.data.euv_frame import load_pec_tables

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ident4")
THETA = dict(dmax=4.0, decay=4.5, v=15.0)
STEP = dict(dmax=0.4, v=1.5)          # 10% 中心差分
C_FRAC, NOISE = 0.03, 0.03
RHO_OUT = np.linspace(0.0, 1.0, 32)

rho_bg, ne_bg, rho_te, te_bg = parse_pp()
pecs = load_pec_tables(C_LINES)
LABELS = list(pecs.keys())
QS = np.array([pecs[k][0] for k in LABELS])


def build_A():
    """A 只依赖 Te/ne 背景（与输运参数无关）→ 全工况复用同一算子。"""
    d = build_case(np.array([0.0, A_LCFS]), np.array([1.0, 1.0]), rho_bg, ne_bg,
                   rho_te, te_bg, pecs, element="c", c_frac=C_FRAC, wl_grid=WL_GRID_VUV)
    return d["A"]


def amp_scale(r, n_tot):
    """物理幅度：同一 c_frac 假设（源项固定）→ 全工况用同一 scale，勿逐工况重归一。"""
    nz = np.interp(RHO_OUT, np.clip(r, 0, A_LCFS) / A_LCFS, n_tot, left=n_tot[0], right=0.0)
    ne = np.interp(RHO_OUT, rho_bg, ne_bg)
    return C_FRAC * ne[RHO_OUT <= 1.0].mean() / max(nz[RHO_OUT <= 1.0].mean(), 1e-300)


def intensities(A, r, n_tot_t, scale):
    """(T, n_line) 绝对线强。"""
    out = []
    for nt in n_tot_t:
        nz = np.interp(RHO_OUT, np.clip(np.array(r), 0, A_LCFS) / A_LCFS,
                       np.array(nt), left=nt[0], right=0.0) * scale
        out.append(A @ nz)
    return np.array(out)


def forward(dmax, v_outer):
    res = run_forward_dv(dmax, THETA["decay"], v_outer, element="'C_'",
                         atomic_weight=12.0, n_time=8, timeout=900)
    if res is None or "n_tot_t" not in res:
        raise RuntimeError(f"forward failed for dmax={dmax} v={v_outer}")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    A = build_A()
    print(f"lines={len(LABELS)} stages={sorted(set(QS.tolist()))}")

    runs = {}
    for tag, dm, v in (("0", THETA["dmax"], THETA["v"]),
                       ("d+", THETA["dmax"] + STEP["dmax"], THETA["v"]),
                       ("d-", THETA["dmax"] - STEP["dmax"], THETA["v"]),
                       ("v+", THETA["dmax"], THETA["v"] + STEP["v"]),
                       ("v-", THETA["dmax"], THETA["v"] - STEP["v"])):
        runs[tag] = forward(dm, v)
        print(f"  run {tag}: ok ({len(runs[tag]['n_tot_t'])} times)")

    scale = amp_scale(runs["0"]["r"], runs["0"]["n_tot_t"][-1])
    I = {tag: intensities(A, r["r"], r["n_tot_t"], scale) for tag, r in runs.items()}
    T = I["0"].shape[0]

    jac_d = (I["d+"] - I["d-"]) / (2 * STEP["dmax"] / THETA["dmax"])    # ∂I/∂(D/D0)
    jac_v = (I["v+"] - I["v-"]) / (2 * STEP["v"] / THETA["v"])           # ∂I/∂(V/V0)

    stages = sorted(set(QS.tolist()))
    mid = stages[len(stages) // 2]
    combos = {}
    for name, sel, times in (("single-steady", np.where(QS == mid)[0], [T - 1]),
                             ("multi-steady", np.arange(len(QS)), [T - 1]),
                             ("single-time", np.where(QS == mid)[0], list(range(T))),
                             ("multi-time", np.arange(len(QS)), list(range(T)))):
        idx = np.ix_(times, sel)
        J = np.column_stack([jac_d[idx].ravel(), jac_v[idx].ravel()])
        sig = np.maximum(NOISE * I["0"][idx].ravel(), 1e-300)
        Jw = J / sig[:, None]
        F = Jw.T @ Jw
        ww = np.linalg.svd(Jw, compute_uv=False)
        rk = int(np.sum(ww > 1e-8 * ww[0]))
        evals, evecs = np.linalg.eigh(F)
        if rk == 2 and evals[0] > 0:
            sigma = np.sqrt(np.diag(np.linalg.inv(F)))
        else:
            sigma = np.array([np.inf, np.inf])
        combos[name] = dict(n_obs=int(J.shape[0]), n_lines=int(len(sel)), n_times=len(times),
                            stage=int(mid), rank=rk,
                            sing_whitened=[float(x) for x in ww],
                            fisher_eig=[float(x) for x in np.sort(evals)[::-1]],
                            cond=float(evals[-1] / evals[0]) if evals[0] > 0 else float("inf"),
                            sigma_D=float(sigma[0]), sigma_V=float(sigma[1]),
                            sigma_D_rel_pct=float(100 * sigma[0]),
                            sigma_V_rel_pct=float(100 * sigma[1]),
                            degenerate_direction=[float(evecs[0, 0]), float(evecs[1, 0])])
        c = combos[name]
        print(f"  {name:14s} n_obs={c['n_obs']:4d} rank={rk} "
              f"sigma_D={c['sigma_D_rel_pct']:8.2f}% sigma_V={c['sigma_V_rel_pct']:8.2f}% "
              f"cond={c['cond']:.3g}")

    # T2-C 验收: 参数精度随观测窗口长度变化（同一批前向，仅取前 T 个时间块）
    window = {}
    for T in (1, 2, 4, 8):
        times = list(range(T))
        idx = np.ix_(times, np.arange(len(QS)))
        J = np.column_stack([jac_d[idx].ravel(), jac_v[idx].ravel()])
        sig = np.maximum(NOISE * I["0"][idx].ravel(), 1e-300)
        F = (J / sig[:, None]).T @ (J / sig[:, None])
        sd = np.sqrt(np.diag(np.linalg.inv(F))) if np.linalg.matrix_rank(F) == 2 else np.array([np.inf, np.inf])
        window[str(T)] = dict(n_obs=int(J.shape[0]), sigma_D_rel_pct=float(100 * sd[0]),
                              sigma_V_rel_pct=float(100 * sd[1]))
        print(f"  window T={T}: sigma_D={100 * sd[0]:6.2f}%  sigma_V={100 * sd[1]:6.2f}%")

    comp = {}
    for k in ("single", "multi"):
        a, b = combos[f"{k}-steady"], combos[f"{k}-time"]
        comp[k] = dict(ratio_sigma_D=a["sigma_D"] / max(b["sigma_D"], 1e-12),
                       ratio_sigma_V=a["sigma_V"] / max(b["sigma_V"], 1e-12),
                       time_better_both=bool(b["sigma_D"] < a["sigma_D"] and b["sigma_V"] < a["sigma_V"]))
    comp["multi_vs_single_steady"] = dict(
        ratio_sigma_D=combos["single-steady"]["sigma_D"] / max(combos["multi-steady"]["sigma_D"], 1e-12),
        ratio_sigma_V=combos["single-steady"]["sigma_V"] / max(combos["multi-steady"]["sigma_V"], 1e-12),
        multi_better_both=bool(combos["multi-steady"]["sigma_D"] < combos["single-steady"]["sigma_D"]
                               and combos["multi-steady"]["sigma_V"] < combos["single-steady"]["sigma_V"]))

    rep = dict(theta=THETA, step=STEP, c_frac=C_FRAC, noise=NOISE, element="C",
               window_vs_time=window,
               n_times=int(T), n_lines=len(LABELS), stages=[int(s) for s in stages],
               selected_stage=int(mid), combos=combos, comparisons=comp)
    path = os.path.join(OUT, "ident4_report.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print("report ->", path)
    return rep


if __name__ == "__main__":
    rep = main()
    c = rep["combos"]
    assert c["multi-time"]["rank"] == 2, "time-resolved multi-stage observation must be full rank"
    assert c["multi-time"]["sigma_V"] < c["multi-steady"]["sigma_V"], \
        "time resolution must improve V (the paper's claim); if this fails the claim is wrong"
    print("PASS: rank(multi-time)=2, sigma_V steady->time "
          f"{c['multi-steady']['sigma_V_rel_pct']:.1f}% -> {c['multi-time']['sigma_V_rel_pct']:.1f}%")
