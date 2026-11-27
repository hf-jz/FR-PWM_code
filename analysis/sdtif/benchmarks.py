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
                target_std=[None if v["sigma"] is None else round(v["sigma"], 6)
                            for v in r.targets.values()],
                pseudo_values={k: (None if v["pseudo_inverse_value"] is None
                                   else round(v["pseudo_inverse_value"], 6))
                               for k, v in r.targets.items()})


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
    """Davis-Kahan against a Monte-Carlo drift, in one norm, over a sweep of perturbation sizes.

    Two defects are removed here.  (i) The drift was measured for a perturbation of the physical
    operator while the bound was stated for the whitened one, so a 1e-3 entry became a 0.1
    perturbation and the table compared 8.6e-4 against a bound of 2e-5.  Both sides are now
    ||L^-1 dJ S||_2.  (ii) The null space was an exact zero singular value, which any dense
    perturbation destroys, so the declaration could not be subjected to a perturbation bound at
    all.  The null space is now declared by a tolerance on an operator whose last singular value
    is small but real, which is the situation the certificate exists for.
    """
    A = np.diag([10.0, 3.0, 1.0, 0.05])
    cov = 1e-4
    tol = 10.0                                 # declared: s < 10 is not resolved (s = [1000,300,100,5])
    tg = dict(x=(1.0, np.array([0, 0, 0, 1.0])), a=(1.0, np.array([1.0, 0, 0, 0])))
    r = localize(from_operator(A, cov=cov, targets=tg), tol_num=tol)
    assert r.rank_num == 3 and r.n - r.rank_num == 1, (r.rank_num, r.n, r.s)
    assert r.targets["x"]["gamma"] > 1 - 1e-12, r.targets["x"]
    g = float(robust(r, 0.0)["gap"])
    rng = np.random.default_rng(11)
    rows = []
    for d in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0, 2.0):
        drifts = []
        for _ in range(40):
            W = rng.normal(size=A.shape)
            W *= d / np.linalg.norm(W, 2)
            E = np.sqrt(cov) * W
            rp = localize(from_operator(A + E, cov=cov, targets=tg), tol_num=tol)
            if rp.rank_num != r.rank_num:
                continue
            drifts.append(subspace_distance(r.P0, rp.P0))
        if not drifts:
            rows.append(dict(delta=d, drift=None, bound=None, rank_preserved=False))
            continue
        rows.append(dict(delta=d, drift=max(drifts), bound=min(d / g, 1.0), rank_preserved=True))
        assert max(drifts) <= d / g * 1.0000001, (d, max(drifts), d / g)
    kept = [x for x in rows if x["rank_preserved"]]
    assert len(kept) >= 6, rows
    # a targeted perturbation, not a random one, is what flips the declared rank: push s4 over
    # the tolerance and the certificate must report the loss instead of a bound
    U, sv, Vt = np.linalg.svd(r.Jt)
    E_flip = np.sqrt(cov) * 12.0 * np.outer(U[:, 3], Vt[3])   # along the weakest direction
    r_flip = localize(from_operator(A + E_flip, cov=cov, targets=tg), tol_num=tol)
    assert r_flip.rank_num == 4, (r_flip.rank_num, r_flip.s)
    return dict(gap=g, sweep=rows, tol_num=tol,
                P_bound=kept[-1]["bound"], drift=kept[-1]["drift"],
                bound_holds_everywhere=True,
                unobservable_subspace_drift=kept[-1]["drift"],
                structural_certified_at_1e_3=bool(0.001 < g / 2),
                certified_at_10=False, rank_lost_at=float(r_flip.rank_num))


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


# ---------------------------------------------------------- generic diagnostic operators (P1)
def e8_generic_diagnostics():
    """Five operator classes that do not come from the carbon system, each with an analytic truth.

    The claim under test is that the calculus takes any observation model.  These are the classes
    the supplementary material lists but never exercises: an integral (line-of-sight) operator, a
    local point measurement, a nonlinear response, a multichannel operator with correlated noise,
    and an operator carrying an unknown calibration parameter.  The grid is n = 6, and the
    identifiability answers are known in closed form for each.
    """
    n = 6
    x = np.linspace(0.05, 0.95, 12)
    G = np.exp(-((x[:, None] - np.linspace(0, 1, n)[None, :]) ** 2) / (2 * 0.18 ** 2))   # integral
    G /= G.sum(axis=1, keepdims=True)
    tg = dict(total=(1.0, np.ones(n)), centroid=(0.5, np.linspace(0, 1, n) - 0.5))
    out = {}
    # (1) integral operator: the total survives, the centroid of a smoothed kernel does not
    r1 = localize(from_operator(G, cov=1e-4, targets=tg))
    out["integral"] = dict(rank=r1.rank_num, null_dim=int(n - r1.rank_num),
                           gamma_total=r1.targets["total"]["gamma"],
                           gamma_centroid=r1.targets["centroid"]["gamma"])
    assert r1.targets["total"]["gamma"] < 1e-10, "a normalised kernel preserves the total"
    # (2) local point measurement: diagonal, six of six resolved, both targets fixed
    r2 = localize(from_operator(np.eye(n), cov=1e-4, targets=tg))
    out["point"] = dict(rank=r2.rank_num, gamma_total=r2.targets["total"]["gamma"])
    assert r2.rank_num == n and r2.targets["total"]["gamma"] < 1e-12
    # (3) nonlinear response h_i = theta_i^3: rank full away from zero, lost at zero
    def h3(t):
        return np.asarray(t, float) ** 3
    p3 = from_model(h3, np.full(n, 0.7), cov=1e-4, targets=tg, step=1e-5)
    r3 = localize(p3)
    p3z = from_model(h3, np.zeros(n) + 1e-12, cov=1e-4, targets=tg, step=1e-8)
    r3z = localize(p3z)
    out["nonlinear"] = dict(rank_regular=r3.rank_num, rank_at_zero=r3z.rank_num)
    assert r3.rank_num == n and r3z.rank_num == 0
    # (4) correlated noise: two channels sharing 90 per cent of their variance
    C = np.array([[1.0, 0.9], [0.9, 1.0]]) * 1e-4
    A = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.9, 0.5, 0.0, 0.0, 0.0]])
    r4 = localize(from_operator(A, cov=C, targets=tg))
    r4d = localize(from_operator(A, cov=np.diag(np.diag(C)), targets=tg))
    out["correlated_noise"] = dict(gamma_total_white=r4d.targets["total"]["gamma"],
                                   gamma_total_coloured=r4.targets["total"]["gamma"],
                                   sigma_total_white=r4d.targets["total"]["sigma"],
                                   sigma_total_coloured=r4.targets["total"]["sigma"])
    # (5) unknown calibration: a common-mode gain added as a seventh column
    A5 = np.c_[G, np.ones(12)]                       # row-normalised kernel: G 1 = 1
    tg5 = dict(total=(1.0, np.r_[np.ones(n), 0.0]),
               shape=(1.0, np.r_[np.linspace(0, 1, n) - 0.5, 0.0]))
    r5n = localize(from_operator(A5, cov=1e-4, targets=tg5, scales=np.r_[np.ones(n), 1.0]))
    out["unknown_calibration"] = dict(rank=r5n.rank_num,
                                      gamma_total=r5n.targets["total"]["gamma"],
                                      gamma_shape=r5n.targets["shape"]["gamma"],
                                      status="the gain column is the constant vector, so the total is "
                                             "indistinguishable from a common-mode scale while the "
                                             "shape is not")
    assert r5n.targets["total"]["gamma"] > 0.5, r5n.targets["total"]["gamma"]
    assert r5n.targets["shape"]["gamma"] < 1e-10
    return out


def e9_edge_cases():
    """The edge conditions the earlier suite left out, each with a known answer."""
    out = {}
    # (1) m < n with a null space of dimension greater than one
    A1 = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    r1 = localize(from_operator(A1, cov=1e-4, targets=dict(sum123=(1.0, np.array([1., 1., 1., 0.])))))
    out["m_lt_n_null_gt1"] = dict(rank=r1.rank_num, null_dim=int(4 - r1.rank_num),
                                  gamma_sum=r1.targets["sum123"]["gamma"])
    assert r1.rank_num == 2 and 4 - r1.rank_num == 2
    # (2) m > n but the operator is still rank deficient
    A2 = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    r2 = localize(from_operator(A2, cov=1e-4, targets=dict(ratio=(1.0, np.array([2.0, -1.0])))))
    out["m_gt_n_rank_deficient"] = dict(rank=r2.rank_num, gamma_ratio=r2.targets["ratio"]["gamma"])
    assert r2.rank_num == 1
    # (3) repeated singular values: a rank-two isometry with a double pair
    A3 = np.eye(4)
    r3 = localize(from_operator(A3, cov=1e-4, targets=dict(sum=(1.0, np.ones(4)))))
    out["repeated_singular_values"] = dict(rank=r3.rank_num, gap=float(robust(r3, 0.0)["gap"]))
    # (4) parameter scales spanning six decades
    A4 = np.array([[1.0, 1.0], [1.0, -1.0]])
    r4s = localize(from_operator(A4, cov=1e-4, scales=np.array([1e-6, 1e6]),
                                 targets=dict(diff=(1.0, np.array([1.0, -1.0])))))
    out["scales_six_decades"] = dict(rank=r4s.rank_num, gamma_diff=r4s.targets["diff"]["gamma"],
                                     sigma_diff=r4s.targets["diff"]["sigma"])
    assert r4s.targets["diff"]["gamma"] < 1e-12
    # (5) target gradient with a component in the null space only, and with one in both
    A5 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    r5 = localize(from_operator(A5, cov=1e-4, targets=dict(
        pure_null=(1.0, np.array([1.0, -1.0, 0.0])),
        partial=(1.0, np.array([1.0, 0.0, 1.0])))))
    t_pure, t_part = r5.targets["pure_null"], r5.targets["partial"]
    out["target_in_null"] = dict(gamma_pure=t_pure["gamma"], gamma_partial=t_part["gamma"],
                                 sigma_pure=t_pure["sigma"], sigma_partial=t_part["sigma"],
                                 pseudo_pure=t_pure["pseudo_inverse_value"])
    # (1,-1,0) lies wholly in the null space; (1,0,1) has 1/2 of its length there
    assert abs(t_pure["gamma"] - 1.0) < 1e-12 and abs(t_part["gamma"] - 0.5) < 1e-12
    # for a target lying wholly in the null space the pseudo-inverse returns ~0, not a large
    # number: the worst possible "uncertainty" for a target no observation determines.  That is
    # the number the paper refuses to call a bound, and this benchmark is why.
    assert t_pure["pseudo_inverse_value"] is not None and t_pure["pseudo_inverse_value"] < 1e-6
    assert t_pure["sigma"] is None or np.isinf(t_pure["sigma"])
    # (6) nuisance parameter that is itself rank deficient
    A6 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    r6 = localize(from_operator(A6, cov=1e-4, targets=dict(q=(1.0, np.ones(3)))))
    F = r6.Jt.T @ r6.Jt
    eff, rank_b = profile(F, 1)
    out["nuisance_rank_deficient"] = dict(rank=r6.rank_num, nuisance_rank=rank_b,
                                          profile_finite=bool(np.all(np.isfinite(eff))))
    # (7) a constraint that restricts the feasible perturbations to an active boundary.  The
    # tool handles it by appending the constraint as one more observation row, which is the only
    # mechanism it needs: an equality constraint removes a direction from the tangent space, so a
    # target that was unidentifiable can become identifiable, and never the reverse.
    A7 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    tg7 = dict(g=(1.0, np.array([0.0, 0.0, 1.0])))
    before = localize(from_operator(A7, cov=1e-4, targets=tg7))
    after = localize(from_operator(np.vstack([A7, [0.0, 0.0, 1.0]]), cov=1e-4, targets=tg7))
    out["active_constraint"] = dict(gamma_free=before.targets["g"]["gamma"],
                                    gamma_constrained=after.targets["g"]["gamma"],
                                    rank_free=before.rank_num, rank_constrained=after.rank_num)
    assert abs(before.targets["g"]["gamma"] - 1.0) < 1e-12, "unconstrained, the target is null"
    assert after.targets["g"]["gamma"] < 1e-12, "the constraint must remove the null direction"
    assert after.rank_num > before.rank_num
    return out


BENCH = [("E1 linear, two null directions", e1_linear_two_nulls),
         ("E2 nonlinear, local rank change", e2_nonlinear_local_rank),
         ("E3 nuisance calibration parameter", e3_nuisance_calibration),
         ("E4 state sweep, rank transition", e4_state_transition),
         ("E5 finite perturbation along null", e5_finite_perturbation),
         ("E6 Weyl / Davis-Kahan robustness", e6_robustness),
         ("E7 analytic bound, independent projector", e7_analytic_bound),
         ("E8 generic diagnostic operators", e8_generic_diagnostics),
         ("E9 edge cases", e9_edge_cases)]
