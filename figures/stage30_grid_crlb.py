"""Does the identifiability grade survive a change of operating point?

Table 2 in the manuscript reports Cramer-Rao bounds for (D, V) at a single operating point, while the
rank evidence is checked on a temperature-density grid. That asymmetry is a fair target: a grade that
flips with Te is not a property of the diagnostic set. This recomputes the bound on a grid by rebuilding
the observation operator at scaled background profiles, with the transport runs reused, and reports where
the grade would change.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library")
sys.path.insert(0, REPO)

from frpwm.data.euv_frame import load_pec_tables                # noqa: E402
from frpwm.data.frame_labels import A_LCFS, C_LINES, WL_GRID_VUV, build_case   # noqa: E402
from frpwm.data.strahl_library import parse_pp                  # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.physics.transport import run_forward_dv              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage30_grid")
THETA = dict(dmax=4.0, decay=4.5, v=15.0)
STEP = dict(dmax=0.4, v=1.5)
C_FRAC, NOISE = 0.03, 0.03
RHO_OUT = np.linspace(0.0, 1.0, 32)
TE_SCALES = (0.5, 1.0, 1.5, 2.0)
NE_SCALES = (0.5, 1.0, 2.0)

rho_bg, ne_bg, rho_te, te_bg = parse_pp()
pecs = load_pec_tables(C_LINES)
LABELS = list(pecs.keys())
QS = np.array([pecs[k][0] for k in LABELS])


def operator(te_scale, ne_scale):
    d = build_case(np.array([0.0, A_LCFS]), np.array([1.0, 1.0]), rho_bg, ne_bg * ne_scale,
                   rho_te, te_bg * te_scale, pecs, element="c", c_frac=C_FRAC, wl_grid=WL_GRID_VUV)
    return d["A"]


def amp_scale(r, n_tot, ne_mean):
    nz = np.interp(RHO_OUT, np.clip(r, 0, A_LCFS) / A_LCFS, n_tot, left=n_tot[0], right=0.0)
    return C_FRAC * ne_mean / max(nz[RHO_OUT <= 1.0].mean(), 1e-300)


def intensities(A, r, n_tot_t, scale):
    return np.array([A @ (np.interp(RHO_OUT, np.clip(np.array(r), 0, A_LCFS) / A_LCFS,
                                    np.array(nt), left=nt[0], right=0.0) * scale) for nt in n_tot_t])


def forward(dmax, v):
    res = run_forward_dv(dmax, THETA["decay"], v, element="'C_'", atomic_weight=12.0,
                         n_time=8, timeout=900)
    if res is None or "n_tot_t" not in res:
        raise RuntimeError("forward failed for dmax=%s v=%s" % (dmax, v))
    return res


def main():
    runs = {}
    for tag, dm, v in (("0", THETA["dmax"], THETA["v"]),
                       ("d+", THETA["dmax"] + STEP["dmax"], THETA["v"]),
                       ("d-", THETA["dmax"] - STEP["dmax"], THETA["v"]),
                       ("v+", THETA["dmax"], THETA["v"] + STEP["v"]),
                       ("v-", THETA["dmax"], THETA["v"] - STEP["v"])):
        runs[tag] = forward(dm, v)
    T = len(runs["0"]["n_tot_t"])
    times = list(range(T))
    all_lines = np.arange(len(QS))

    report = {"theta": THETA, "step": STEP, "noise": NOISE,
              "te_scales": list(TE_SCALES), "ne_scales": list(NE_SCALES),
              "grid": {}, "note": "multi-time observation set (four lines, eight time slices)"}

    sD, sV = [], []
    for te in TE_SCALES:
        for ne in NE_SCALES:
            A = operator(te, ne)
            ne_mean = float(np.interp(RHO_OUT, rho_bg, ne_bg * ne)[RHO_OUT <= 1.0].mean())
            scale = amp_scale(runs["0"]["r"], runs["0"]["n_tot_t"][-1], ne_mean)
            I = {tag: intensities(A, r["r"], r["n_tot_t"], scale) for tag, r in runs.items()}
            jd = (I["d+"] - I["d-"]) / (2 * STEP["dmax"] / THETA["dmax"])
            jv = (I["v+"] - I["v-"]) / (2 * STEP["v"] / THETA["v"])
            idx = np.ix_(times, all_lines)
            J = np.column_stack([jd[idx].ravel(), jv[idx].ravel()])
            sig = np.maximum(NOISE * I["0"][idx].ravel(), 1e-300)
            Jw = J / sig[:, None]
            F = Jw.T @ Jw
            w = np.linalg.svd(Jw, compute_uv=False)
            rank = int(np.sum(w > 1e-8 * w[0]))
            ev = np.linalg.eigvalsh(F)
            if rank == 2 and ev[0] > 0:
                sd, sv = np.sqrt(np.diag(np.linalg.inv(F)))
            else:
                sd = sv = np.inf
            key = "Te x%.1f, ne x%.1f" % (te, ne)
            report["grid"][key] = dict(rank=rank, cond=float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
                                       sigma_D_rel_pct=float(100 * sd), sigma_V_rel_pct=float(100 * sv))
            sD.append(100 * sd)
            sV.append(100 * sv)
            print("%-20s rank=%d  sigma_D=%7.2f%%  sigma_V=%7.2f%%" % (key, rank, 100 * sd, 100 * sv))

    # the density scaling turns out not to reach the operator: record it rather than imply a two-dimensional sweep
    report["density_invariant"] = True
    report["summary"] = {
        "sigma_D_rel_pct": {"min": float(min(sD)), "max": float(max(sD))},
        "sigma_V_rel_pct": {"min": float(min(sV)), "max": float(max(sV))},
        "rank_min": min(v["rank"] for v in report["grid"].values()),
        "grade_flips_at_12pct": [k for k, v in report["grid"].items() if v["sigma_V_rel_pct"] <= 12.0],
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "grid_crlb.json"), "w"), indent=1)
    print("\ndensity scaling does not change the operator; the sweep is in temperature")
    print("bound on V: %.1f to %.1f per cent across the grid; rank %d everywhere"
          % (min(sV), max(sV), report["summary"]["rank_min"]))
    print("points where the V bound falls to or below the 12 per cent line:",
          report["summary"]["grade_flips_at_12pct"] or "none")


if __name__ == "__main__":
    main()
