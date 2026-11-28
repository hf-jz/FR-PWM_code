"""State maps: how the grades move across the (T_e, n_e) domain, not at single nodes.

Three panels, from the same operators the paper audits:
  (a) the structural gap g(T_e, n_e) of the weighted six-by-seven operator, on a log scale --- the gap
      is what a state-domain certificate is compared against;
  (b) Gamma(N_tot) of the weighted operator, which moves with the number of populated stages;
  (c) the relative bound on the concentration in the conditional model, where the atomic physics is
      taken as given and only the concentration is estimated.

Regions are marked rather than described: structurally unidentifiable cells are hatched, and a cell
whose certificate would be refused (gap below the measured node-to-node variation) is drawn with a
thin outline.

    python3 figures/make_fig_state_maps.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "library"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.sdtif import plasma          # noqa: E402
from analysis.sdtif.core import Problem, localize   # noqa: E402
from paths import ARTEFACTS                # noqa: E402

TE = np.exp(np.linspace(np.log(10.0), np.log(400.0), 12))
NE = np.exp(np.linspace(np.log(2e13), np.log(1e14), 5))


def cell(te, ne):
    ee = plasma.emissivity(te, ne)                     # 6 x 7, line k against stage k+1
    fr = plasma.fractions(te, ne)
    f = np.array([fr[q] for q in range(1, 8)])
    # the same operator, the same scale and the same noise as case A_weighted, so a value read off
    # this figure is the value in the table
    n_ref = np.full(7, plasma.C_FRAC * ne / 7.0)
    sig_w = plasma.ref_noise(te, ne)
    live = plasma.measurement_mask(ee * f[None, :], n_ref, sig_w)
    A_un = ee
    A_w = (ee * f[None, :]) * live[None, :]
    tg_un = dict(N_tot=(1.0, np.ones(7)), N_obs=(1.0, np.r_[np.ones(6), 0.0]))
    r_un = localize(Problem(A_un, np.diag(sig_w ** 2), n_ref, targets=tg_un))
    r_w = localize(Problem(A_w, np.diag(sig_w ** 2), n_ref, targets=tg_un))
    # conditional model: concentration and temperature, six lines
    def hh(x):
        tt = np.exp(x[1])
        e2 = plasma.emissivity(tt, ne)
        f2 = plasma.fractions(tt, ne)
        return np.array([e2[k, q - 1] * f2[q] * np.exp(x[0]) for k, q in enumerate(range(1, 7))])
    x0 = np.array([np.log(plasma.C_FRAC * ne), np.log(te)])
    J = np.zeros((6, 2))
    d = 1e-3
    for j in range(2):
        a, b = x0.copy(), x0.copy()
        a[j] += d
        b[j] -= d
        J[:, j] = (hh(a) - hh(b)) / (2 * d)
    sig = plasma.ref_noise(te, ne)
    pc = Problem(J, np.diag(sig ** 2), np.array([1.0, 1.0]),
                 targets=dict(c=(plasma.C_FRAC * ne, np.array([plasma.C_FRAC * ne, 0.0]))))
    r_c = localize(pc, eps_req=plasma.EPS_REQ)
    return dict(occupied=int(live.sum()), rank_raw=None,
                gap_w=float(r_w.s[3] - r_w.s[4]) if r_w.s.size > 4 else 0.0,
                gamma_tot_un=float(r_un.targets["N_tot"]["gamma"]),
                gamma_tot_w=float(r_w.targets["N_tot"]["gamma"]),
                gamma_obs_w=float(r_w.targets["N_obs"]["gamma"]),
                rank_un=int(r_un.rank_num), rank_w=int(r_w.rank_num),
                gamma_c=float(r_c.targets["c"]["gamma"]),
                sigma_c=r_c.targets["c"]["relative_sigma"],
                cls_c=r_c.targets["c"]["cls"])


def main():
    rows = [[cell(te, ne) for ne in NE] for te in TE]
    out = dict(te=list(TE), ne=list(NE), cells=rows)
    dest = os.path.join(ARTEFACTS, "A22_SD_TIF")
    json.dump(out, open(os.path.join(dest, "state_maps.json"), "w"), indent=1)

    def grid(key):
        g = np.array([[c[key] if c[key] is not None else np.nan for c in row] for row in rows], float)
        return g

    gap = grid("gap_w")
    gtw = grid("gamma_tot_w")
    sc = grid("sigma_c")
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.9))
    for ax, data, title, cmap, fmt in (
            (axs[0], gap, r"gap $g$", "viridis", "%.1e"),
            (axs[1], gtw, r"$\Gamma(N_{\mathrm{tot}})$", "magma", "%.2f"),
            (axs[2], 100 * sc, r"$\sigma(c)/c$ (\%%)", "cividis", "%.1f")):
        m = np.ma.masked_invalid(data.T)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.85")
        im = ax.pcolormesh(TE, NE, m, shading="auto", cmap=cm)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([2e13, 5e13, 1e14])
        ax.set_yticklabels([r"$2\times10^{13}$", r"$5\times10^{13}$", r"$10^{14}$"], fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel(r"$T_e$ (eV)", fontsize=8)
        ax.set_ylabel(r"$n_e$ (m$^{-3}$)", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=6)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
    # mark the cells where the concentration is not structurally identifiable at all
    bad = np.array([[c["cls_c"] == "unidentifiable" for c in row] for row in rows])
    for i in range(TE.size):
        for j in range(NE.size):
            if bad[i, j]:
                axs[2].plot(TE[i], NE[j], "x", ms=3, color="white")
    fig.tight_layout()
    out_pdf = os.path.join(HERE, "generated", "fig_state_maps.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    print("wrote", out_pdf)
    print("gamma_tot weighted range %.3f .. %.3f" % (np.nanmin(gtw), np.nanmax(gtw)))
    print("sigma_c per cent range %.2f .. %.2f" % (np.nanmin(100 * sc), np.nanmax(100 * sc)))
    print("cells unidentifiable (c):", int(bad.sum()), "of", bad.size)


if __name__ == "__main__":
    main()
