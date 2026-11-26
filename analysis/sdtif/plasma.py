"""The plasma cases, computed.  Nothing in the output is typed in by hand.

Case A  unconditional inversion: seven free charge-state densities, six lines.
Case B  conditional inversion: (total concentration, T_e) with populations from the
        ionisation balance -- the temperature dependence of identifiability lives here.
Case C  three lines plus an interferometer with an unknown radiometric gain.
Case D  steady impurity transport with an unmeasured source amplitude.
Case E  the state map over the T_e x n_e grid, with a box certificate on Gamma.

Writes sdtif_out/sdtif_cases.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "library"))
from paths import ROOT as _PKGROOT, ARTEFACTS, LIBRARY, OUT  # noqa: E402
from .core import (Problem, from_model, from_operator, localize, refine_axis,  # noqa: E402
                   robust, systematic_box)

LINES6 = ("C I 834.4", "C II 426.6", "C III 288.4", "C IV 222.3", "C V 33.4", "C VI 182.2")
STAGES = ("C I", "C II", "C III", "C IV", "C V", "C VI", "C VII")
C_FRAC = 0.01
NOISE = 0.03
EPS_REQ = 0.10                     # required accuracy on a unit dimensionless perturbation
TE_GRID = (10.0, 15.0, 25.0, 30.0, 40.0, 60.0, 90.0, 130.0, 200.0, 400.0)
NE_GRID = (2e13, 1e14)


def _adas():
    from frpwm.physics.adas import LINES, PecCache, ion_fractions
    return LINES, PecCache(), ion_fractions


_L = None


def _atomic():
    """Atomic data on first use: the real-data case must run without an ADAS release."""
    global _L
    if _L is None:
        _L = _adas()
    return _L


def fractions(te, ne):
    """Ionisation balance, keyed by charge state 1..7 (C I .. C VII)."""
    _, _, ionf = _atomic()
    return {int(z) + 1: float(v) for z, v in ionf("c", ne, te)}


def emissivity(te, ne):
    """Line emissivity per unit density of the emitting stage [arb.].  Diagonal by design:
    each line sees one charge state, and no line of the fully stripped stage exists."""
    lines, pc, _ = _atomic()
    e = np.zeros((len(LINES6), 7))
    for k, nm in enumerate(LINES6):
        el, q, wl = lines[nm]
        e[k, q - 1] = float(pc.pec_value(el, q, wl, ne, te)) * ne
    return e


def ref_noise(te, ne):
    """One-sigma line noise, evaluated at the equal-population reference state.

    The noise is relative to the reference brightness rather than to the brightness at the
    node being analysed: a line whose stage has vanished at that node then contributes no
    information (a zero row) instead of an infinite one (a zero divided into a finite
    signal), which is the difference between a well-posed operator and an artefact.
    """
    e = emissivity(te, ne)
    return np.maximum(NOISE * (e @ np.full(7, C_FRAC * ne / 7.0)), 1e-30)


def targets_unconditional(n_ref):
    tot = n_ref.sum()
    qv = np.arange(1, 8, dtype=float)
    zbar = float((qv * n_ref).sum() / tot)
    return dict(N_tot=(tot, np.ones(7)),
                N_obs=(n_ref[:6].sum(), np.array([1.0] * 6 + [0.0])),
                mean_charge=(zbar, (qv - zbar) / tot))


# ------------------------------------------------------------------------------ case A
def case_A(nodes=None):
    nodes = nodes or [(te, ne) for te in TE_GRID for ne in NE_GRID]
    rows, results = [], []
    for te, ne in nodes:
        A = emissivity(te, ne)
        n_ref = np.full(7, C_FRAC * ne / 7.0)
        y = A @ n_ref
        sig = np.maximum(NOISE * y, 1e-30)
        p = from_operator(A, cov=np.diag(sig ** 2), scales=n_ref.copy(),
                          targets=targets_unconditional(n_ref), names=STAGES,
                          meta=dict(te=te, ne=ne, mode="unconditional"))
        r = localize(p, eps_req=EPS_REQ)
        results.append(((te, ne), r))
        rows.append(dict(te=te, ne=ne, rank_num=r.rank_num, rank_prac=r.rank_prac,
                         singular_values=[float(x) for x in r.s],
                         null_basis=[[float(x) for x in c] for c in r.V0.T],
                         null_dim=int(7 - r.rank_num),
                         targets={k: dict(gamma=v["gamma"], cls=v["cls"],
                                          relative_sigma=v["relative_sigma"]) for k, v in r.targets.items()},
                         selfcheck=r.selfcheck, certified=robust(r, 1e-8)["certified"]))
    return rows, results


# ------------------------------------------------------------------------------ case B
def case_B():
    """Conditional inversion: theta = (ln c, ln T_e); populations follow the balance.

    With T_e known this is the six-parameter inversion of the paper; with T_e free it is a
    two-parameter inversion in which the line ratios carry the temperature.  The target is
    the total carbon concentration, which is identifiable in both -- to what accuracy is
    the question, and it is the answer that changes across the grid.
    """
    rows = []
    for te in TE_GRID:
        for ne in NE_GRID:
            c_ref = C_FRAC * ne
            fr = fractions(te, ne)

            def h(x, te0=te, ne0=ne):
                t, n = np.exp(x[1]), ne0
                f = fractions(t, n)
                return np.array([emissivity(t, n)[k, q - 1] * f[q] * np.exp(x[0])
                                 for k, q in enumerate(range(1, 7))])
            J = np.zeros((6, 2))
            d = 1e-3
            h0 = h([np.log(c_ref), np.log(te)])
            for j in range(2):
                e = np.zeros(2)
                e[j] = d
                J[:, j] = (h([np.log(c_ref) + e[0], np.log(te) + e[1]]) -
                           h([np.log(c_ref) - e[0], np.log(te) - e[1]])) / (2 * d)
            sig = ref_noise(te, ne)
            tg = dict(c=(c_ref, np.array([c_ref, 0.0])))
            r_free = localize(from_operator(J, cov=np.diag(sig ** 2), scales=[1.0, 1.0],
                                            targets=tg), eps_req=EPS_REQ)
            r_fix = localize(from_operator(J[:, :1], cov=np.diag(sig ** 2), scales=[1.0],
                                           targets=dict(c=(c_ref, np.array([c_ref])))), eps_req=EPS_REQ)
            rows.append(dict(te=te, ne=ne, rank_num=r_free.rank_num, rank_prac=r_free.rank_prac,
                             singular_values=[float(x) for x in r_free.s],
                             gamma_c=r_free.targets["c"]["gamma"],
                             cls=r_free.targets["c"]["cls"],
                             rel_sigma_te_free=r_free.targets["c"]["relative_sigma"],
                             rel_sigma_te_known=r_fix.targets["c"]["relative_sigma"],
                             min_fraction=float(min(fr.values())),
                             dominant=int(max(fr, key=fr.get))))
    return rows


# ------------------------------------------------------------------------------ case C
def case_C():
    """Three lines + interferometer, unknown radiometric gain (see module docstring)."""
    te, ne = 60.0, 1e14
    sel = ("C IV 222.3", "C V 33.4", "C VI 182.2")

    def eps(lne, lte):
        """The three selected lines per unit carbon concentration at (ln n_e, ln T_e)."""
        n, t = np.exp(lne), np.exp(lte)
        f = fractions(t, n)
        e = emissivity(t, n)
        return np.array([e[3, 3] * f[4], e[4, 4] * f[5], e[5, 5] * f[6]])

    th0 = np.array([np.log(ne), np.log(te)])
    d = 1e-3
    e0 = eps(*th0)
    col_ne = (eps(th0[0] + d, th0[1]) - eps(th0[0] - d, th0[1])) / (2 * d)
    col_te = (eps(th0[0], th0[1] + d) - eps(th0[0], th0[1] - d)) / (2 * d)
    J3 = np.column_stack([col_ne, col_te, e0])          # columns: ln n_e, ln T_e, ln g
    scales = np.array([1.0, 1.0, 0.05])
    tg = dict(n_e=(ne, np.array([ne, 0.0, 0.0])), T_e=(te, np.array([0.0, te, 0.0])))
    no_int = localize(from_operator(J3, cov=NOISE ** 2, scales=scales, targets=tg), eps_req=EPS_REQ)
    J4 = np.vstack([J3, np.array([[1.0, 0.0, 0.0]])])   # interferometer, no gain in the path
    with_int = localize(from_operator(J4, cov=np.diag(np.r_[np.full(3, NOISE ** 2), 0.02 ** 2]),
                                      scales=scales, targets=tg), eps_req=EPS_REQ)
    return dict(singular_values_lines_only=[float(x) for x in no_int.s],
                rank_lines_only=no_int.rank_num, rank_prac_lines_only=no_int.rank_prac,
                null_basis_lines_only=[[float(x) for x in c] for c in no_int.V0.T],
                gamma_lines_only={k: v["gamma"] for k, v in no_int.targets.items()},
                cls_lines_only={k: v["cls"] for k, v in no_int.targets.items()},
                reason_lines_only={k: v["reason"] for k, v in no_int.targets.items()},
                singular_values_with_interferometer=[float(x) for x in with_int.s],
                rank_with_interferometer=with_int.rank_num,
                rank_prac_with_interferometer=with_int.rank_prac,
                gamma_with_interferometer={k: v["gamma"] for k, v in with_int.targets.items()},
                cls_with_interferometer={k: v["cls"] for k, v in with_int.targets.items()},
                reason_with_interferometer={k: v["reason"] for k, v in with_int.targets.items()},
                relative_sigma_with_interferometer={k: v["relative_sigma"]
                                                    for k, v in with_int.targets.items()})


# ------------------------------------------------------------------------------ case D
def case_D():
    """Steady transport with an unmeasured source amplitude: exact (D,V,S) -> lambda(...).

    Equation (ss) of the manuscript has no sink term, so the map is exactly invariant under
    the joint scaling; the discretised solver is checked against that statement rather than
    assumed to obey it.
    """
    from frpwm.invert.dv_inversion import transport_solve, par_D, par_V
    rho = np.linspace(0.02, 1.0, 40)
    Vp = 4 * np.pi ** 2 * 1.7 * (0.6 * rho) ** 2 / 0.6 + 1.0
    S = np.exp(-((rho - 0.92) / 0.05) ** 2)
    D0, V0, S0, VREF, TAU = 1.0, -1.5, 1.0, 10.0, 1e30      # no sink: eq (ss)

    def prof(th):
        return transport_solve(par_D(rho, np.exp(th[0])), par_V(rho, th[1] * VREF),
                               np.exp(th[2]) * S, TAU, rho, Vp, n_bc=0.0)

    ch = np.linspace(0.1, 0.9, 8)
    h = lambda th: np.log(np.maximum(np.interp(ch, rho, prof(th)), 1e-300))
    th0 = np.array([np.log(D0), V0 / VREF, np.log(S0)])
    tg = dict(D0=(D0, np.array([D0, 0.0, 0.0])),
              V0=(abs(V0), np.array([0.0, VREF, 0.0])),
              source_amplitude=(S0, np.array([0.0, 0.0, S0])),
              D_over_V=(D0 / abs(V0), np.array([1.0, -1.0 / (V0 / VREF), 0.0])))
    p = from_model(h, th0, cov=NOISE ** 2, scales=np.array([1.0, V0 / VREF, 1.0]),
                   targets=tg, step=1e-4)
    r = localize(p, eps_req=EPS_REQ)
    lam = 1.0 + 1e-4
    sym = float(np.max(np.abs(prof(th0 + np.log(lam) * np.array([1.0, 1.0, 1.0])) - prof(th0)))
                / np.max(np.abs(prof(th0))))
    return dict(singular_values=[float(x) for x in r.s], rank_num=r.rank_num,
                rank_prac=r.rank_prac, tol_num=r.tol_num,
                null_basis=[[float(x) for x in c] for c in r.V0.T],
                symmetry_residual=sym, fd_converged=p.meta.get("fd_converged"),
                targets={k: dict(gamma=v["gamma"], cls=v["cls"],
                                 relative_sigma=v["relative_sigma"]) for k, v in r.targets.items()},
                selfcheck=r.selfcheck)


# ------------------------------------------------------------------------ case E the map
def case_E(te_grid=None, ne_grid=None):
    """Cross-state analysis: the conditional operator, a refinement-limited error bound and
    a box certificate.

    A Lipschitz bound on J over a continuum cannot be established on a coarse grid, so the
    object reported here is the measured variation between adjacent nodes, refined until the
    variation is small against the singular-value gap.  Where that target is not met the node
    is reported as not certified; the tool never upgrades an uncertified node to a grade.
    """
    te_grid = te_grid or TE_GRID
    ne_grid = ne_grid or NE_GRID

    def node(te, ne):
        c_ref = C_FRAC * ne

        def hh(x):
            tt, nn = np.exp(x[1]), ne
            ff = fractions(tt, nn)
            ee = emissivity(tt, nn)
            return np.array([ee[k, q - 1] * ff[q] * np.exp(x[0]) for k, q in enumerate(range(1, 7))])

        x0 = np.array([np.log(c_ref), np.log(te)])
        d = 1e-3
        J = np.zeros((6, 2))
        for j in range(2):
            e1, e2 = x0.copy(), x0.copy()
            e1[j] += d
            e2[j] -= d
            J[:, j] = (hh(e1) - hh(e2)) / (2 * d)
        sig = ref_noise(np.exp(x0[1]), ne)
        p = Problem(J, np.diag(sig ** 2), np.array([1.0, 1.0]),
                    targets=dict(c=(c_ref, np.array([c_ref, 0.0]))))
        return (J / sig[:, None]), localize(p, eps_req=EPS_REQ)

    def sweep(te_list, ne):
        """Adjacent-node variation along T_e, and the certificate at each node."""
        ops, res = zip(*[node(t, ne) for t in te_list])
        rows = []
        for i, t in enumerate(te_list):
            nb = [np.linalg.norm(ops[i] - ops[j], 2) for j in (i - 1, i + 1) if 0 <= j < len(ops)]
            delta = float(max(nb)) if nb else float("inf")
            rb = robust(res[i], delta, ["c"])
            rows.append(dict(te=t, ne=ne, delta_J=delta, gap=rb["gap"],
                             certified=bool(rb["certified"]),
                             gamma=res[i].targets["c"]["gamma"],
                             rel_sigma=res[i].targets["c"]["relative_sigma"],
                             order=rb["rank_prac_upper"]))
        return rows

    coarse = sweep(list(TE_GRID), 1e14)
    fine_te = list(np.exp(np.linspace(np.log(10.0), np.log(400.0), 41)))
    fine = sweep(fine_te, 1e14)
    # across density at fixed temperature, coarse then refined: the refinement is what buys a
    # certificate, and the count of certified nodes is the acceptance criterion
    def density_sweep(ne_list):
        ops_n, res_n = zip(*[node(60.0, n) for n in ne_list])
        out = []
        for i, n in enumerate(ne_list):
            nb = [np.linalg.norm(ops_n[i] - ops_n[j], 2) for j in (i - 1, i + 1) if 0 <= j < len(ops_n)]
            delta = float(max(nb)) if nb else float("inf")
            rb = robust(res_n[i], delta, ["c"])
            out.append(dict(ne=n, delta_J=delta, gap=rb["gap"], certified=bool(rb["certified"]),
                            rel_sigma=res_n[i].targets["c"]["relative_sigma"]))
        return out

    dens_coarse = density_sweep([2e13, 5e13, 1e14])
    dens_fine = density_sweep(list(np.exp(np.linspace(np.log(2e13), np.log(1e14), 12))))
    return dict(coarse=coarse, fine=fine, density_coarse=dens_coarse, density_fine=dens_fine,
                n_certified_coarse=int(sum(r["certified"] for r in coarse)),
                n_certified_fine=int(sum(r["certified"] for r in fine)),
                n_certified_density_coarse=int(sum(r["certified"] for r in dens_coarse)),
                n_certified_density_fine=int(sum(r["certified"] for r in dens_fine)),
                delta_density_max_coarse=float(max(r["delta_J"] for r in dens_coarse)),
                delta_density_max_fine=float(max(r["delta_J"] for r in dens_fine)),
                delta_max_coarse=float(max(r["delta_J"] for r in coarse)),
                delta_max_fine=float(max(r["delta_J"] for r in fine)),
                gamma_max=float(max(r["gamma"] for r in coarse + fine)),
                rel_sigma_min=float(min(999.0 if r["rel_sigma"] is None else r["rel_sigma"] for r in coarse + fine)),
                rel_sigma_max=float(max(999.0 if r["rel_sigma"] is None else r["rel_sigma"] for r in coarse + fine)))


# ------------------------------------------------------------------------------ case A2
def case_A_weighted():
    """The same six lines with the ionisation balance folded into the columns.

    Column $k$ is then emissivity times the equilibrium fraction of stage $k$, the unknown is a
    scale factor on each stage's population, and a stage whose population vanishes contributes a
    zero column.  This is the operator whose per-node rank was tabulated in earlier drafts; it is
    reported here next to case A because the two answer different questions, and the difference
    is exactly what produced the apparent contradiction between the rank table and the null
    vector.
    """
    rows = []
    for te in TE_GRID:
        for ne in NE_GRID:
            A0 = emissivity(te, ne)
            fr = fractions(te, ne)
            fv = np.array([fr.get(q, 0.0) for q in range(1, 8)])
            A = A0 * fv[None, :]
            n_ref = np.full(7, C_FRAC * ne / 7.0)
            sig = ref_noise(te, ne)
            p = from_operator(A, cov=np.diag(sig ** 2), scales=n_ref.copy(),
                              targets=targets_unconditional(n_ref), names=STAGES,
                              meta=dict(te=te, ne=ne, mode="equilibrium_weighted"))
            r = localize(p, eps_req=EPS_REQ)
            rows.append(dict(te=te, ne=ne, rank_num=r.rank_num, rank_prac=r.rank_prac,
                             singular_values=[float(x) for x in r.s],
                             null_basis=[[float(x) for x in c] for c in r.V0.T],
                             occupied=int(np.sum(fv > 1e-12)),
                             targets={k: dict(gamma=v["gamma"], cls=v["cls"],
                                              relative_sigma=v["relative_sigma"])
                                      for k, v in r.targets.items()}))
    return rows


# ------------------------------------------------------- case F real device data (MAST CXRS)
def _audit_real_shot(shot):
    """One MAST shot: twelve CXRS radii, unknown channel calibration, amplitude against shape.

    Nothing here is simulated except the parametric profile used to localise the analysis.  The
    noise is the measured radial roughness of the profile --- this release carries no per-point
    error bar --- and it is labelled as an estimate wherever it is reported.
    """
    d = np.load(os.path.join(ARTEFACTS, "A22_SD_TIF", "inputs", "level2_%s.npz" % shot))
    R, tt, ti = d["major_radius"], d["time"], d["t_i"]
    # the frame used is the one carrying the most usable radii, not a fixed time: the shots of this
    # campaign do not share a flat-top window and three of them have no CXRS at 0.125 s at all
    ok_all = np.isfinite(ti) & (ti > 0) & (R > 0.75)[:, None]
    per_frame = ok_all.sum(axis=0)
    j = int(np.argmax(per_frame))
    y, x = ti[:, j], R
    ok = np.isfinite(y) & (y > 0) & (R > 0.75)
    if ok.sum() < 12:
        raise ValueError("shot %s: %d usable radii at t = %.3f s, need at least 12"
                         % (shot, int(ok.sum()), tt[j]))
    x, y = x[ok], y[ok]
    rho = (x - 0.85) / 0.85                                  # normalised, reference rho = 0.85
    sel = np.linspace(0, rho.size - 1, 12).round().astype(int)   # twelve analysis radii
    xr, yr = rho[sel], y[sel]
    # noise: roughness of the measured profile, sqrt(pi/6) * RMS of second differences
    d2 = np.diff(yr, 2)
    sig = np.full(yr.size, max(np.sqrt(np.pi / 6) * float(np.sqrt(np.mean(d2 ** 2))), 0.03 * yr.max()))
    # local fit of ln T = ln g + ln T0 - (a x + b x^2), gains shared, shapes free
    A = np.c_[np.ones(xr.size), -xr, -xr ** 2]
    coef, *_ = np.linalg.lstsq(A / sig[:, None], np.log(yr) / sig, rcond=None)
    lnT0, a, b = coef
    th0 = np.array([np.log(1.0), lnT0, a, b])                 # (ln g, ln T0, a, b_intercept)

    def h(th):
        return np.exp(th[0] + th[1] - (th[2] * xr + th[3] * xr ** 2))

    def dlnL(xq):
        """Gradient of ln of the gradient length L = 1/(a+2bx) at x_q."""
        den = th0[2] + 2 * th0[3] * xq
        g = np.zeros(4)
        g[2], g[3] = -1.0 / den, -2 * xq / den
        return g

    tg = dict(T_absolute=(float(np.interp(0.0, xr, yr)), np.array([0.0, 1.0, 0.0, 0.0])),
              ln_L_T=(0.0, dlnL(0.0)),
              T_shape_coeff=(a, np.array([0.0, 0.0, 1.0, 0.0])))
    p = from_model(h, th0, cov=np.diag(sig ** 2), scales=np.array([1.0, 1.0, 1.0, 1.0]),
                   targets=tg, step=1e-5)
    r = localize(p, eps_req=EPS_REQ)
    fit = float(np.max(np.abs(h(th0) - yr) / sig))
    sb = systematic_box(r, 0.15)                             # 15 per cent calibration error
    # how far the noise may grow before the gradient-length grade is lost (module M7)
    def certify(lo, hi):
        mu = hi
        pm = from_model(h, th0, cov=np.diag((mu * sig) ** 2), targets=tg, step=1e-5)
        rm = localize(pm, eps_req=EPS_REQ)
        t = rm.targets["ln_L_T"]
        s_t = t["sigma"]
        return dict(certified=bool(t["gamma"] < 1e-10 and s_t is not None and s_t <= EPS_REQ),
                    delta_J=t["gamma"])
    sweep = refine_axis(certify, 1.0, 64.0, depth=5)
    return dict(shot=int(d["shot"]), t_index=int(j), time=float(tt[j]), n_radii=int(yr.size),
                x_range=[float(xr.min()), float(xr.max())],
                T_range=[float(yr.min()), float(yr.max())], sigma_pct=float(100 * np.median(sig / yr)),
                fit_max_sigma=fit, rank_num=r.rank_num, rank_prac=r.rank_prac,
                null_basis=[[float(v) for v in c] for c in r.V0.T],
                targets={k: dict(gamma=v["gamma"], cls=v["cls"], reason=v["reason"],
                                 sigma=v["sigma"], relative_sigma=v["relative_sigma"])
                         for k, v in r.targets.items()},
                calibration_box=dict(rel=sb["rel"], delta_J=sb["delta_J_rigorous"], targets=sb["targets"]),
                noise_sweep=dict(certified=sweep["certified"], refused=sweep["refused"],
                                 evaluations=sweep["evaluations"],
                                 certified_measure=sweep["certified_measure"]),
                selfcheck=r.selfcheck)


def case_F_realdata(shots=None):
    """The same audit on every archived MAST shot, so the grade is a map and not one point.

    Reference shot detailed (28315, the one with a time slice at 0.125 s); the rest contribute the
    four numbers that decide whether the grade is a property of the diagnostic or of the state:
    Gamma of the amplitude, Gamma of the gradient length, its sigma and the noise multiplier at
    which it would be refused.  If those four hold across the campaign, the verdict travels with
    the diagnostic; where they do not, the state node is where it stops.
    """
    dest = os.path.join(ARTEFACTS, "A22_SD_TIF", "inputs")
    have = sorted(int(f.split("_")[1].split(".")[0]) for f in os.listdir(dest)
                  if f.startswith("level2_") and f.endswith(".npz"))
    shots = shots or have
    ref = _audit_real_shot(28315 if 28315 in shots else shots[0])
    rows = []
    for sh in shots:
        a = ref if sh == ref["shot"] else _audit_real_shot(sh)
        gT = a["targets"]["T_absolute"]["gamma"]
        gL = a["targets"]["ln_L_T"]["gamma"]
        cert = a["noise_sweep"]["certified"]
        rows.append(dict(shot=a["shot"], time=a["time"], n_radii=a["n_radii"],
                         T_range=a["T_range"], sigma_pct=a["sigma_pct"],
                         fit_max_sigma=a["fit_max_sigma"], rank_num=a["rank_num"],
                         gamma_T=gT, gamma_L=gL,
                         sigma_L=a["targets"]["ln_L_T"]["sigma"],
                         calibration_verdict_L=a["calibration_box"]["targets"]["ln_L_T"]["verdict"],
                         certified_mu=float(max(b for _, b in cert)) if cert else 0.0))
    ref = dict(ref)
    ref["map"] = rows
    ref["map_summary"] = dict(
        n_shots=len(rows),
        amplitude_degenerate=sum(1 for r in rows if abs(r["gamma_T"] - 2 ** -0.5) < 1e-6),
        shape_identifiable=sum(1 for r in rows if r["gamma_L"] < 1e-10),
        shape_certified_at_nominal=sum(1 for r in rows if r["certified_mu"] >= 1.0),
        shape_stable_under_calibration=sum(1 for r in rows if r["calibration_verdict_L"] == "stable"),
        sigma_L_range=[min(r["sigma_L"] for r in rows if r["sigma_L"]),
                       max(r["sigma_L"] for r in rows if r["sigma_L"])],
        fit_worst=max(r["fit_max_sigma"] for r in rows),
        sigma_pct_range=[min(r["sigma_pct"] for r in rows), max(r["sigma_pct"] for r in rows)])
    return ref


def main():
    dest = os.path.join(ARTEFACTS, "A22_SD_TIF")
    os.makedirs(dest, exist_ok=True)
    rows_a, results_a = case_A()
    rows_b = case_B()
    out = dict(settings=dict(noise=NOISE, carbon_fraction=C_FRAC, eps_req=EPS_REQ,
                             te_grid=list(TE_GRID), ne_grid=list(NE_GRID),
                             line_set=list(LINES6), stages=list(STAGES),
                             real_data="MAST 28315 level2 CXRS, t=0.125 s"),
               case_A_unconditional=rows_a, case_A_weighted=case_A_weighted(),
               case_B_conditional=rows_b,
               case_C_gain=case_C(), case_D_transport=case_D())
    out["case_E_map"] = case_E()
    out["case_F_realdata"] = case_F_realdata()
    with open(os.path.join(dest, "cases.json"), "w") as f:
        json.dump(out, f, indent=1)

    a = rows_a[0]
    print("A unconditional: rank %d/%d  null %d  s=%s" % (
        a["rank_num"], a["rank_prac"], a["null_dim"],
        np.array2string(np.array(a["singular_values"]), precision=3)))
    print("   Gamma %s" % {k: round(v["gamma"], 4) for k, v in a["targets"].items()})
    print("   N_obs relative sigma %s" % [round(r["targets"]["N_obs"]["relative_sigma"], 4) if
                                          r["targets"]["N_obs"]["relative_sigma"] else None
                                          for r in rows_a[:4]])
    print("B conditional: Gamma(c) %s" % sorted({round(r["gamma_c"], 6) for r in rows_b}))
    ks = [999.0 if r["rel_sigma_te_known"] is None else r["rel_sigma_te_known"] for r in rows_b]
    fs = [999.0 if r["rel_sigma_te_free"] is None else r["rel_sigma_te_free"] for r in rows_b]
    print("   sigma_c (Te known) %.3f .. %.3f  (Te free) %.3f .. %.3f" % (
        min(ks), max(ks), min(fs), max(fs)))
    print("   rank_num %s rank_prac %s" % (sorted({r["rank_num"] for r in rows_b}),
                                           sorted({r["rank_prac"] for r in rows_b})))
    c = out["case_C_gain"]
    print("C gain: rank %d -> %d  Gamma %s -> %s" % (
        c["rank_lines_only"], c["rank_with_interferometer"],
        {k: round(v, 3) for k, v in c["gamma_lines_only"].items()},
        {k: round(v, 3) for k, v in c["gamma_with_interferometer"].items()}))
    d = out["case_D_transport"]
    print("D transport: s %s rank %d/%d symmetry %.2e Gamma %s" % (
        np.round(d["singular_values"], 4), d["rank_num"], d["rank_prac"], d["symmetry_residual"],
        {k: round(v["gamma"], 3) for k, v in d["targets"].items()}))
    e = out["case_E_map"]
    print("E map: certified %d/%d coarse, %d/%d fine | delta_J max %.1f -> %.1f | Gamma_max %.1e"
          % (e["n_certified_coarse"], len(e["coarse"]), e["n_certified_fine"], len(e["fine"]),
             e["delta_max_coarse"], e["delta_max_fine"], e["gamma_max"]))
    print("   density axis: certified %d/%d coarse, %d/%d fine (delta %.1f -> %.1f)"
          % (e["n_certified_density_coarse"], len(e["density_coarse"]),
             e["n_certified_density_fine"], len(e["density_fine"]),
             e["delta_density_max_coarse"], e["delta_density_max_fine"]))
    print("   rel sigma c: %.4f .. %.4f" % (e["rel_sigma_min"], e["rel_sigma_max"]))
    return out


if __name__ == "__main__":
    main()
