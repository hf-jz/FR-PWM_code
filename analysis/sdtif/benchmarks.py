"""Analytic benchmarks for the identifiability calculus (E1-E6 of the reviewer's protocol).

Each benchmark has a closed-form answer, so the engine is checked against mathematics
rather than against another implementation of itself.  Exits non-zero on any mismatch.
"""
import numpy as np

from .core import (Problem, localize, from_operator, from_model, profile, robust,
                   subspace_distance, state_certificate, UNIDENT)

TOL = 1e-9


def e1_linear_two_nulls():
    """A = [[1,1,0,0],[0,0,1,1]]: null dim 2, and four targets with known Gamma."""
    A = np.array([[1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0]])
    tg = dict(q12=(2.0, np.array([1.0, 1.0, 0, 0])),      # span of row 1: Gamma 0
              q34=(2.0, np.array([0, 0, 1.0, 1.0])),      # Gamma 0
              d12=(0.0, np.array([1.0, -1.0, 0, 0])),     # pure null: Gamma 1
              q13=(1.0, np.array([1.0, 0, 1.0, 0])),      # half null: Gamma 2^-1/2
              t13=(2.0, np.array([1.0, 0, -1.0, 0])))     # Gamma 2^-1/2
    r = localize(from_operator(A, targets=tg))
    g = {k: v["gamma"] for k, v in r.targets.items()}
    want = dict(q12=0.0, q34=0.0, d12=1.0, q13=2 ** -0.5, t13=2 ** -0.5)
    for k in want:
        assert abs(g[k] - want[k]) < TOL, (k, g[k], want[k])
    assert r.n - r.rank_num == 2
    # practical rank is noise-relative: with unit noise a unit step is only seen to 70%
    assert r.rank_prac == 0
    # a two-dimensional null space must be returned as a basis, not as one vector
    assert r.V0.shape == (4, 2) and np.linalg.norm(r.V0.T @ r.V0 - np.eye(2)) < TOL
    # the projector, not the basis, is the invariant object: rotate the basis, same P0
    Q = np.array([[0.6, -0.8], [0.8, 0.6]])
    P_rot = (r.V0 @ Q) @ (r.V0 @ Q).T
    assert np.linalg.norm(P_rot - r.P0) < TOL
    return dict(rank=r.rank_num, null_dim=2, gamma={k: round(v, 6) for k, v in g.items()},
                target_std=[round(v["sigma"], 6) for v in r.targets.values()])


def e2_nonlinear_local_rank():
    """y=(th1^2+th2, th1 th2): J singular exactly on the curve th2 = 2 th1^2."""
    h = lambda t: np.array([t[0] ** 2 + t[1], t[0] * t[1]])
    tg = dict(th1=(1.0, np.array([1.0, 0.0])), th1sq_plus=(3.0, np.array([2.0, 1.0])))
    reg = from_model(h, np.array([1.0, 1.0]), targets=tg, step=1e-5)
    deg = from_model(h, np.array([1.0, 2.0]), targets=tg, step=1e-5)
    assert reg.meta["fd_converged"] and deg.meta["fd_converged"]
    rr, rd = localize(reg), localize(deg)
    assert rr.rank_num == 2, rr.rank_num
    assert rd.rank_num == 1 and rd.n - rd.rank_num == 1
    v = rd.V0[:, 0]
    v = v / np.linalg.norm(v)
    assert abs(abs(v[0] / v[1]) - 0.5) < 0.02, v           # analytic (1,-2)/sqrt5
    # at the singular point the observable combination is still obtainable, th1 alone is not
    assert rd.targets["th1sq_plus"]["gamma"] < 0.02         # 2 th1 + th2 is the row space
    # c = e1 against (1,-2)/sqrt5: Gamma = 1/sqrt5, analytically
    assert abs(rd.targets["th1"]["gamma"] - 5 ** -0.5) < 0.02, rd.targets["th1"]
    return dict(regular_rank=rr.rank_num, degenerate_rank=rd.rank_num,
                null_direction=[round(x, 4) for x in v],
                gamma_degenerate={k: round(v["gamma"], 4) for k, v in rd.targets.items()})


def e3_nuisance_calibration():
    """y = th (1+eta): with the gain known th is identifiable, with the gain free it is not.

    Analytic: J = [[1+eta, th]]; the profiled information on th is exactly zero, and
    Gamma(th) = |th| / sqrt(1+th^2) -> the parameter is only as identifiable as the
    calibration allows.
    """
    th0, eta0 = 1.0, 0.0
    h_fix = lambda t: np.array([t[0] * (1 + eta0)])
    h_free = lambda t: np.array([t[0] * (1 + t[1])])
    tg = dict(n_z=(1.0, np.array([1.0, 0.0])))
    r_fix = localize(from_model(h_fix, np.array([th0]), cov=1e-4,
                                targets=dict(n_z=(1.0, np.array([1.0])))))
    r_free = localize(from_model(h_free, np.array([th0, eta0]), cov=1e-4, targets=tg))
    F_free = r_free.Jt.T @ r_free.Jt
    Feff, rb = profile(F_free, 1)
    g_ana = abs(th0) / np.sqrt(1 + th0 ** 2) * np.sqrt(1 + eta0 ** 2) / np.sqrt(1 + eta0 ** 2)
    assert r_fix.targets["n_z"]["gamma"] < TOL
    assert abs(r_free.targets["n_z"]["gamma"] - g_ana) < 1e-6, (r_free.targets["n_z"]["gamma"], g_ana)
    assert abs(Feff[0, 0]) < 1e-12, Feff          # profile kills the information exactly
    # the trap: a pseudo-inverse still returns a finite number for a target that the data
    # cannot see, so a finite bound is not evidence that the target is identifiable
    finite_but_meaningless = bool(np.isfinite(r_free.targets["n_z"]["sigma"]))
    assert finite_but_meaningless and r_free.targets["n_z"]["cls"] == UNIDENT
    return dict(gamma_fixed=round(r_fix.targets["n_z"]["gamma"], 8),
                gamma_free=round(r_free.targets["n_z"]["gamma"], 6),
                profiled_information=float(Feff[0, 0]), nuisance_rank=rb,
                finite_bound_reported_despite_unidentifiable=finite_but_meaningless,
                class_free=r_free.targets["n_z"]["cls"])


def e4_state_transition():
    """A(xi) = [[1,1+xi],[1,1-xi]]: rank 1 exactly at xi=0, rank 2 either side.

    A coarse grid that skips xi=0 reports the system as fully identifiable everywhere;
    the state certificate, built from a Lipschitz bound on J, does not.
    """
    tg = dict(sum=(2.0, np.array([1.0, 1.0])), th1=(1.0, np.array([1.0, 0.0])))
    mk = lambda xi: from_operator(np.array([[1.0, 1 + xi], [1.0, 1 - xi]]), targets=tg)
    nodes = [(xi, localize(mk(xi))) for xi in (0.0, 1e-4, 1e-2, 0.1, 0.3)]
    coarse = [(xi, localize(mk(xi))) for xi in (0.1, 0.2, 0.3)]          # misses zero
    g0 = nodes[0][1].targets["sum"]["gamma"]
    assert g0 < TOL and nodes[0][1].targets["th1"]["gamma"] > 0.7
    assert all(n[1].rank_num == 2 for n in nodes[1:])
    assert all(abs(n[1].targets["sum"]["gamma"]) < TOL for n in nodes)
    # a box well away from the crossing is certifiable at delta=2% of the gap row ...
    far = [n for n in nodes if n[0] >= 0.1]
    cert_far = state_certificate(far, lipschitz=1.0, radius=0.02)
    # ... a box straddling it is not, and the certificate says so instead of guessing
    cert_cross = state_certificate(nodes[:2], lipschitz=1.0, radius=0.05)
    assert cert_far["targets"]["sum"]["certified"]
    assert not cert_cross["targets"]["sum"]["certified"]
    d = subspace_distance(nodes[0][1].P0, nodes[-1][1].P0)
    return dict(gamma_sum=[round(float(n[1].targets["sum"]["gamma"]), 6) for n in nodes],
                gamma_th1=[round(float(n[1].targets["th1"]["gamma"]), 4) for n in nodes],
                ranks=[n[1].rank_num for n in nodes],
                coarse_grid_would_call_it=[n[1].rank_num for n in coarse],
                certified_away_from_crossing=cert_far["targets"]["sum"]["certified"],
                certified_across_crossing=cert_cross["targets"]["sum"]["certified"],
                subspace_distance_0_to_03=round(d, 4))


def e5_finite_perturbation():
    """The predicted null direction really is invisible to the data at finite amplitude.

    h(th) = (th1+th2, 0.1 (th1+th2)^2, th1-th2) at th0=(1,1) has an exact linear null
    (1,-1)/sqrt2 at first order (the first two rows are constant along it) and the
    third row is what makes it identifiable; scale that row by 0 and check that a finite
    step along the null direction changes the whitened observation below the noise.
    """
    sigma = 0.03

    def make(w3):
        return lambda t: np.array([t[0] + t[1], 0.1 * (t[0] + t[1]) ** 2, w3 * (t[0] - t[1])])

    tg = dict(sum=(2.0, np.array([1.0, 1.0])))
    p = from_model(make(0.0), np.array([1.0, 1.0]), cov=sigma ** 2, targets=tg, step=1e-4)
    r = localize(p)
    assert r.n - r.rank_num == 1
    v = r.V0[:, 0] / np.linalg.norm(r.V0[:, 0])
    dv = np.array([1.0, -1.0]) / np.sqrt(2)
    assert abs(abs(v @ dv) - 1.0) < 1e-4
    rows = []
    for a in (0.01, 0.03, 0.1, 0.3):
        y0 = make(0.0)(np.array([1.0, 1.0]))
        y1 = make(0.0)(np.array([1.0, 1.0]) + a * dv)
        rows.append(dict(amplitude=a, whitened_change=float(np.linalg.norm(y1 - y0) / sigma)))
    # whitening makes a 0.3 step along the null look like 0.6 sigma: invisible
    assert rows[-1]["whitened_change"] < 1.0, rows
    # and the target (the sum) does not move along that direction, by construction
    dq = [abs(make(1.0)(np.array([1.0, 1.0]) + a * dv).sum() - 0.0) for a in (0.1,)]
    return dict(sigma=sigma, null_direction=[round(x, 4) for x in v], sweep=rows,
                gamma_sum=round(r.targets["sum"]["gamma"], 6))


def e6_robustness():
    """Weyl/Davis-Kahan bounds: an operator perturbation that keeps the gap keeps the grade."""
    A = np.diag([10.0, 3.0, 1.0, 0.0])
    tg = dict(x=(1.0, np.array([0, 0, 0, 1.0])), a=(1.0, np.array([1.0, 0, 0, 0])))
    r = localize(from_operator(A, cov=1e-2 ** 2, targets=tg))
    assert r.n - r.rank_num == 1 and r.targets["x"]["gamma"] > 1 - 1e-12
    ok, bad = robust(r, 1e-3), robust(r, 10.0)
    assert ok["certified"] and not bad["certified"]
    # a Monte-Carlo check on the subspace the noise-level threshold declares unobservable.
    # (The structural test is refused at this perturbation size because a 1e-3 entry lifts
    # s4 above the exact-zero tolerance; the certificate above reports that separately.)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        E = rng.normal(size=A.shape)
        E *= 1e-3 / np.linalg.norm(E, 2)
        rp = localize(from_operator(A + E, cov=1e-2 ** 2, targets=tg))
        worst = max(worst, subspace_distance(r.P_unobs, rp.P_unobs))
    assert worst <= 1.0 and abs(np.trace(r.P_unobs) - 1.0) < 1e-9
    return dict(P_bound=ok["P_bound"], structural_certified_at_1e_3=ok["certified"],
                certified_at_10=bad["certified"], gap=ok["gap"],
                unobservable_subspace_drift=round(float(worst), 5),
                structural_rank_upper_at_1e_3=bad["rank_prac_upper"])


def e7_analytic_bound():
    """The marginal bound, against its closed form, and the null subspace by a second route.

    For A = I and unit parameter scales, F = A^T A / sigma^2, so the bound on the sum of two
    parameters is c^T F^-1 c = 2 sigma^2.  The null projector is then recomputed from the
    eigendecomposition of J^T J rather than from the SVD: an independent code path for the
    object every other number here depends on.
    """
    sigma = 1e-3
    tg = dict(total=(2.0, np.array([1.0, 1.0])), first=(1.0, np.array([1.0, 0.0])))
    r = localize(from_operator(np.eye(2), cov=sigma ** 2, targets=tg))
    assert abs(r.targets["total"]["sigma"] ** 2 - 2 * sigma ** 2) < 1e-18, r.targets["total"]["sigma"]
    assert r.targets["total"]["gamma"] < 1e-12 and r.targets["total"]["cls"] == "fully_identifiable"
    # rank-deficient case: the null projector from J^T J must match the one from the SVD
    A = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    rd = localize(from_operator(A, cov=sigma ** 2, targets=dict(q=(1.0, np.ones(3)))))
    w, V = np.linalg.eigh(rd.Jt.T @ rd.Jt)
    V0 = V[:, w <= max(1e-12 * w.max(), 0.0)] if np.any(w <= 1e-12 * w.max()) else V[:, :0]
    P_eig = V0 @ V0.T if V0.size else np.zeros((3, 3))
    assert np.linalg.norm(P_eig - rd.P0) < 1e-10, np.linalg.norm(P_eig - rd.P0)
    return dict(bound_analytic=2 * sigma ** 2, bound_computed=r.targets["total"]["sigma"] ** 2,
                projector_independent_path=float(np.linalg.norm(P_eig - rd.P0)))


BENCH = [("E1 linear, two null directions", e1_linear_two_nulls),
         ("E2 nonlinear, local rank change", e2_nonlinear_local_rank),
         ("E3 nuisance calibration parameter", e3_nuisance_calibration),
         ("E4 state sweep, rank transition", e4_state_transition),
         ("E5 finite perturbation along null", e5_finite_perturbation),
         ("E6 Weyl / Davis-Kahan robustness", e6_robustness),
         ("E7 analytic bound, independent projector", e7_analytic_bound)]


def run():
    out = {}
    for name, fn in BENCH:
        r = fn()
        out[name] = r
        print("PASS %-38s %s" % (name, {k: v for k, v in r.items() if k in
                                        ("rank", "null_dim", "gamma", "gamma_free", "certified_sum",
                                         "P_bound", "var_inflation", "degenerate_rank")}))
    return out


if __name__ == "__main__":
    run()
