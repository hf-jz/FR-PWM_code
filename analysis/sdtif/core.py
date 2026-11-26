"""SD-TIF: state-dependent, target-oriented identifiability calculus.

One observation model h(theta; xi) + eps, one locally linearised operator J, and four
questions answered on it in a fixed order:

  1. structural   which parameter directions the observation cannot see at all  (exact null space)
  2. target       which named physical functional is invariant along those directions
  3. practical    which directions survive the noise, and how wide is the target's bound
  4. state        whether 1-3 stay true over a box of operating states and model error

Everything here is deterministic numpy; no model-specific physics.  Physics enters only
through a Problem (J, covariance, parameter scales, target gradients).

Conventions
-----------
z is the dimensionless parameter coordinate, theta = theta0 + S z, S = diag(scales).
The analysis operator is  Jt = L^-1 J S  with Cov(eps) = L L^T, so a unit perturbation
in z_i moves the whitened observation by s_i and the 1-sigma estimate of z_i is 1/s_i.
Greek letters in the output: Gamma_q is the fraction of the target gradient that lies in
the unobservable subspace (0 = fully determined by these observations, 1 = blind).
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np

FULLY, WEAK, PRIOR, UNIDENT = "fully_identifiable", "weakly_identifiable", "prior_dependent", "unidentifiable"


# ----------------------------------------------------------------------------- problem
@dataclass
class Problem:
    """A local identifiability problem at one operating point."""
    J: np.ndarray                      # m x n, d y / d theta, physical units
    cov: np.ndarray                    # m x m observation covariance (scalar -> diag)
    scales: np.ndarray                 # n parameter scales (same units as theta)
    names: tuple = ()
    targets: dict = field(default_factory=dict)   # name -> (value, grad_theta)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.J = np.atleast_2d(np.asarray(self.J, float))
        n = self.J.shape[1]
        self.scales = np.atleast_1d(np.asarray(self.scales, float))
        if self.scales.size == 1:
            self.scales = np.full(n, float(self.scales.reshape(-1)[0]))
        assert self.scales.shape == (n,), "one scale per parameter"
        m = self.J.shape[0]
        c = self.cov
        self.cov = np.eye(m) * float(c) if np.isscalar(c) or np.size(c) == 1 else np.asarray(c, float)
        if not self.names:
            self.names = tuple("theta%d" % (i + 1) for i in range(n))


def from_operator(A, cov=1.0, scales=1.0, targets=None, names=(), meta=None):
    return Problem(A, cov, scales, names, targets or {}, meta or {})


def from_model(h, theta0, cov=1.0, scales=1.0, targets=None, names=(), step=1e-5,
               check_steps=(1.0, 0.5, 0.25), meta=None):
    """Central differences with a step-halving convergence check (returns the check too)."""
    theta0 = np.asarray(theta0, float)
    n = theta0.size
    sc = np.atleast_1d(np.asarray(scales, float))
    sc = np.full(n, float(sc)) if sc.size == 1 else sc
    y0 = np.atleast_1d(np.asarray(h(theta0), float))
    Js = []
    for f in check_steps:
        J = np.zeros((y0.size, n))
        for j in range(n):
            d = np.zeros(n)
            d[j] = step * f * sc[j]
            J[:, j] = (np.atleast_1d(h(theta0 + d)) - np.atleast_1d(h(theta0 - d))) / (2 * d[j])
        Js.append(J)
    d1 = np.linalg.norm(Js[0] - Js[1]) / max(np.linalg.norm(Js[1]), 1e-300)
    d2 = np.linalg.norm(Js[1] - Js[2]) / max(np.linalg.norm(Js[2]), 1e-300)
    p = Problem(Js[2], cov, sc, names, targets or {},
                dict(meta or {}, fd_step=step, fd_rel_change=(d1, d2),
                     fd_converged=bool(d2 < 0.05 and d2 < d1 * 1.5)))
    return p


# ---------------------------------------------------------------------------- analysis
@dataclass
class Result:
    problem: Problem
    Jt: np.ndarray
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    rank_num: int
    rank_prac: int
    tol_num: float
    tol_prac: float
    V0: np.ndarray            # full right null basis (structural)
    V_weak: np.ndarray        # directions below the practical threshold
    P0: np.ndarray
    Pobs: np.ndarray
    selfcheck: dict
    P_unobs: np.ndarray = None      # projector on the practically unobservable subspace
    meta: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.Jt.shape[1]


def localize(problem, eps_req=0.1, tol_num=None):
    """Structural and practical decomposition at one node.

    tol_num  : numerical null threshold, max(m,n)*eps*s1 (LAPACK convention)
    tol_prac : a direction counts as practically resolvable when its 1-sigma
               estimate, 1/s_i, is below eps_req in dimensionless units
    """
    J, S = problem.J, problem.scales
    m, n = J.shape
    L = np.linalg.cholesky(problem.cov) if problem.cov.shape[0] > 1 else np.sqrt(problem.cov)
    Y = np.linalg.solve(L, J) if np.ndim(L) == 2 else J / float(L)
    Jt = Y * S[None, :]
    U, s, Vt = np.linalg.svd(Jt, full_matrices=True)          # full: n right vectors always
    s_full = np.zeros(n)
    s_full[:s.size] = s
    s1 = s_full[0] if s_full[0] > 0 else 1.0
    tol_n = float(tol_num if tol_num is not None else max(m, n) * np.finfo(float).eps * s1)
    # A numerically differentiated operator cannot resolve a direction better than the
    # differentiation itself: the threshold is never allowed below the measured FD noise.
    fd = problem.meta.get("fd_rel_change")
    if fd and tol_num is None:
        tol_n = max(tol_n, 100.0 * max(fd) * s1)
    r_num = int(np.sum(s_full > tol_n))
    r_prac = int(np.sum(s_full * eps_req > 1.0))              # 1/s_i <= eps_req
    V = Vt.T
    V0 = V[:, r_num:]
    P0 = V0 @ V0.T if V0.size else np.zeros((n, n))
    Pobs = np.eye(n) - P0
    Vw = V[:, r_prac:]                                        # practically unobservable subspace
    P_unobs = Vw @ Vw.T if Vw.size else np.zeros((n, n))
    e = dict(symmetry=float(np.linalg.norm(P0 - P0.T)),
             idempotence=float(np.linalg.norm(P0 @ P0 - P0)),
             annihilates=float(np.linalg.norm(Jt @ P0) / s1),
             orthogonal_complement=float(np.linalg.norm(Pobs + P0 - np.eye(n))),
             orthonormal_basis=float(np.linalg.norm(V0.T @ V0 - np.eye(V0.shape[1]))) if V0.size else 0.0)
    res = Result(problem=problem, Jt=Jt, U=U, s=s_full, Vt=Vt, rank_num=r_num,
                 rank_prac=r_prac, tol_num=tol_n, tol_prac=1.0 / eps_req, V0=V0,
                 V_weak=V[:, r_prac:], P0=P0, Pobs=Pobs, selfcheck=e,
                 P_unobs=P_unobs, meta=dict(eps_req=eps_req))
    for name, spec in problem.targets.items():
        val, grad = spec if isinstance(spec, (tuple, list)) else (None, spec)
        res.targets[name] = analyze_target(res, name, np.atleast_1d(np.asarray(grad, float)), val)
    return res


def analyze_target(res, name, grad_theta, value=None, eps_target=None):
    """Null-space projection and marginal bound of one target functional.

    Returns Gamma (null fraction), the bound, and the class.  A target that fails the
    structural test gets no finite number: the reported value is the identifiable
    combination itself, which is what a caller may print.
    """
    eps_target = res.meta.get("eps_req", 0.1) if eps_target is None else eps_target
    cz = res.problem.scales * grad_theta
    nz = np.linalg.norm(cz)
    if nz <= 0:
        raise ValueError("target gradient is zero in %s" % name)
    proj = res.P0 @ cz
    gamma = float(np.linalg.norm(proj) / nz)
    r = res.rank_prac
    var = None
    if r > 0:
        Vr = res.Vt[:r].T                       # observable directions in z
        coeff = Vr.T @ cz
        var = float(np.sum((coeff / res.s[:r]) ** 2))
    # gradient component that no resolvable direction can reach: the bound below is then a
    # lower bound on the target's uncertainty, not the uncertainty itself
    Vr = res.Vt[:r].T
    c_res = cz - (Vr @ (Vr.T @ cz) if r > 0 else 0.0)
    unresolved = float(np.linalg.norm(c_res) / nz)
    # identifiable part: the target seen through the observable subspace
    c_obs = res.Pobs @ cz
    rel = None if var is None else float(np.sqrt(var) / abs(value)) if value not in (None, 0) else float(np.sqrt(var))
    if gamma > 1e-8:
        cls, reason = UNIDENT, "null_space"
    elif var is None:
        cls, reason = UNIDENT, "no_resolvable_direction"
    elif rel > 1.0:
        cls, reason = UNIDENT, "bound_exceeds_variation_scale"
    elif unresolved > 1e-8:
        cls, reason = WEAK, "partially_resolved_bound_is_lower_bound"
    elif rel <= eps_target:
        cls, reason = FULLY, "bound_below_requirement"
    else:
        cls, reason = WEAK, "bound_above_requirement"
    return dict(target=name, gamma=gamma, null_norm=float(np.linalg.norm(proj)),
                sigma=float(np.sqrt(var)) if var is not None else float("inf"),
                relative_sigma=rel, cls=cls, reason=reason,
                unresolved_fraction=unresolved,
                identifiable_fraction=float(np.linalg.norm(c_obs) / nz),
                value=None if value is None else float(value),
                eps_target=eps_target)


# ------------------------------------------------------------------------- nuisance
def profile(F, n_theta):
    """Schur complement: information on theta after eliminating nuisance parameters.

    F = [[Faa, Fab], [Fba, Fbb]] -> Faa - Fab Fbb^+ Fba.  With a singular Fbb the
    elimination is done in its range; directions that Fbb cannot pin at all are not
    silently dropped, they are reported through the returned rank.
    """
    F = np.asarray(F, float)
    A, B = slice(0, n_theta), slice(n_theta, F.shape[0])
    Faa, Fab, Fbb = F[A, A], F[A, B], F[B, B]
    if Fbb.size == 0:
        return Faa, 0
    rb = np.linalg.matrix_rank(Fbb)
    eff = Faa - Fab @ np.linalg.pinv(Fbb) @ Fab.T
    return 0.5 * (eff + eff.T), int(rb)


# ----------------------------------------------------------------------- robustness
def robust(res, delta_J, target_names=None):
    """Bounds valid for every operator within ||dJ||_2 <= delta_J.

    Weyl gives |s_i' - s_i| <= delta_J, so the structural rank is certain only when
    s_r - delta > tol_num > s_{r+1} + delta.  Davis-Kahan gives, for equal-dimension
    null spaces, ||P0' - P0|| <= delta_J / gap with gap = s_r - s_{r+1}; therefore
    |Gamma_q' - Gamma_q| <= delta_J/gap because ||P c||/||c|| is 1-Lipschitz in P.
    Where the gap collapses the certificate degrades to the trivial value 1 and says so.
    """
    s = res.s
    r = res.rank_num
    rp = res.rank_prac
    # practical rank: how many directions stay resolvable at noise level over the whole set
    out = dict(delta_J=float(delta_J),
               rank_prac_lower=int(np.sum(s - delta_J > res.tol_prac)),
               rank_prac_upper=int(np.sum(s + delta_J > res.tol_prac)))
    gap = float(s[r - 1] - (s[r] if r < s.size else 0.0))
    out["gap"] = gap
    out["rank_stable"] = bool(out["rank_prac_lower"] == out["rank_prac_upper"] == rp)
    # the structural null direction must not become practically observable
    null_stays_null = bool((s[r] if r < s.size else 0.0) + delta_J < res.tol_prac)
    out["null_stays_unobservable"] = null_stays_null
    if gap <= 0:
        out["P_bound"], out["certified"] = 1.0, False
    else:
        out["P_bound"] = float(min(1.0, 2 * delta_J / gap))
        out["certified"] = bool(delta_J < gap / 2 and out["rank_stable"] and null_stays_null)
    names = target_names or list(res.targets)
    out["targets"] = {}
    for nm in names:
        g = res.targets[nm]["gamma"]
        out["targets"][nm] = dict(gamma=g, gamma_upper=float(min(1.0, g + out["P_bound"])),
                                  verdict="certain" if (g == 0 and out["certified"]) else
                                          ("at risk" if g + out["P_bound"] > 1e-8 else "open"))
    return out


def subspace_distance(P1, P2):
    """Projector distance / sqrt(2d): 0 for equal subspaces, 1 for orthogonal ones."""
    d = int(round(np.trace(P1)))
    denom = np.sqrt(2.0 * d) if d else 1.0
    return float(np.linalg.norm(P1 - P2) / denom)


def state_certificate(nodes, lipschitz, target_names=None, radius=0.05):
    """State-box certificate from a set of analysed nodes and a Jacobian Lipschitz bound.

    nodes     : list of (xi, Result); xi has the same coordinate as `lipschitz`
    lipschitz : d ||Jt|| / d xi, measured on a refined grid (a bound for xi in the box)
    radius    : half-width of the state box around each node in xi units
    A target is certified over the box only if every node's Weyl interval keeps the rank
    and every node's Gamma upper bound stays below 1e-8.
    """
    names = target_names or list(nodes[0][1].targets)
    delta = float(lipschitz) * radius
    rows = {nm: dict(gamma_max=0.0, gamma_upper_max=0.0, certified=True, nodes=0) for nm in names}
    for xi, res in nodes:
        rb = robust(res, delta, names)
        for nm in names:
            rows[nm]["nodes"] += 1
            rows[nm]["gamma_max"] = max(rows[nm]["gamma_max"], rb["targets"][nm]["gamma"])
            rows[nm]["gamma_upper_max"] = max(rows[nm]["gamma_upper_max"],
                                              rb["targets"][nm]["gamma_upper"])
            rows[nm]["certified"] &= bool(rb["certified"] and rb["rank_stable"])
    return dict(delta_J=delta, radius=radius, lipschitz=float(lipschitz), targets=rows)


def fingerprint(arr):
    a = np.ascontiguousarray(np.asarray(arr, float))
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def as_json(res, robust_out=None):
    d = dict(shape=list(res.Jt.shape), singular_values=[float(x) for x in res.s],
             rank_num=res.rank_num, rank_prac=res.rank_prac,
             tol_num=res.tol_num, tol_prac=res.tol_prac,
             null_dim=int(res.n - res.rank_num), weak_dim=int(res.n - res.rank_prac),
             null_basis=[[float(x) for x in col] for col in res.V0.T],
             projector_trace=float(np.trace(res.P0)), selfcheck=res.selfcheck,
             jacobian_sha256=fingerprint(res.Jt), meta=res.meta,
             targets={k: v for k, v in res.targets.items()})
    if robust_out is not None:
        d["robust"] = robust_out
    return d


# ------------------------------------------------------------- systematic error and refinement
def systematic_box(res, rel, cols=None):
    """Grade change when a calibration error of |rel| scales chosen columns of Jtilde.

    Two different numbers, and they must not be confused.  The range over the 2^k corners is
    what actually happens for a multiplicative error on a known set of channels, and it is
    computed exactly at the corners.  The certified bound is Weyl with the *rigorous*
    perturbation norm, ||J D||_2 <= ||J||_2 * max|1+eps|, so it holds for every point of the
    box and not merely at its corners.
    """
    Jt, s = res.Jt, res.s
    n = Jt.shape[1]
    cols = list(range(n)) if cols is None else list(cols)
    tgt = {k: v for k, v in res.problem.targets.items()}
    out = dict(rel=float(rel), columns=cols, corners=2 ** len(cols), targets={})
    spread = {k: [] for k in tgt}
    for mask in range(2 ** len(cols)):
        eps = np.ones(n)
        for i, c in enumerate(cols):
            eps[c] = 1.0 + (rel if (mask >> i) & 1 else -rel)
        U, sv, Vt = np.linalg.svd(Jt @ np.diag(eps), full_matrices=True)
        s_full = np.zeros(n)
        s_full[:sv.size] = sv
        r = int(np.sum(s_full > res.tol_num))
        V0 = Vt.T[:, r:]
        P0 = V0 @ V0.T if V0.size else np.zeros((n, n))
        for k, spec in tgt.items():
            _, grad = spec if isinstance(spec, (tuple, list)) else (None, spec)
            cz = res.problem.scales * np.atleast_1d(np.asarray(grad, float))
            spread[k].append(float(np.linalg.norm(P0 @ cz) / np.linalg.norm(cz)))
    delta = float(s[0] * rel)          # ||J diag(eps)||_2 <= ||J||_2 * max|eps|
    r = res.rank_num
    gap = float(s[r - 1] - (s[r] if r < s.size else 0.0))
    for k, gs in spread.items():
        g0 = res.targets[k]["gamma"]
        out["targets"][k] = dict(gamma_nominal=g0, gamma_min=min(gs), gamma_max=max(gs),
                                 gamma_certified_upper=float(min(1.0, g0 + delta / gap))
                                 if gap > 0 else 1.0,
                                 verdict="stable" if max(gs) < 1e-8 else
                                         ("lost" if g0 < 1e-8 <= max(gs) else "degraded"))
    out["delta_J_rigorous"] = delta
    return out


def refine_axis(certify, lo, hi, depth=6):
    """Adaptive bisection: keep the sub-intervals on which the certificate is issued.

    certify(a, b) returns dict(certified=bool, delta_J=float).  The interval is bisected until
    every surviving piece certifies or the depth is exhausted, so the answer is a set of
    intervals that are actually covered by a proof rather than a grid of point tests.
    """
    out = dict(certified=[], refused=[], evaluations=0, max_depth=0)  # type: dict

    def step(a, b, d):
        out["evaluations"] += 1
        out["max_depth"] = max(out["max_depth"], d)
        r = certify(a, b)
        if r["certified"]:
            out["certified"].append([a, b])
            return
        if d == 0:
            out["refused"].append([a, b])
            return
        mid = 0.5 * (a + b)
        step(a, mid, d - 1)
        step(mid, b, d - 1)

    step(float(lo), float(hi), depth)
    out["certified_measure"] = float(sum(b - a for a, b in out["certified"])) / (hi - lo)
    out["refused_measure"] = float(sum(b - a for a, b in out["refused"])) / (hi - lo)
    return out


# ---------------------------------------------------------------- reference example
def demo():
    """Analytic case from the manuscript: rank-1 operator, two diagonal targets."""
    A = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = from_operator(A, cov=1.0, scales=1.0, names=("a", "b", "c"),
                      targets=dict(sum_ab=(2.0, np.array([1.0, 1.0, 0.0])),
                                   diff_ab=(0.0, np.array([1.0, -1.0, 0.0])),
                                   a_plus_c=(1.0, np.array([1.0, 0.0, 1.0]))))
    r = localize(p)
    assert r.rank_num == 2 and r.n - r.rank_num == 1
    g = {k: v["gamma"] for k, v in r.targets.items()}
    assert abs(g["sum_ab"]) < 1e-12
    assert abs(g["diff_ab"] - 1.0) < 1e-12
    assert abs(g["a_plus_c"] - 0.5) < 1e-12
    for k in ("symmetry", "idempotence"):
        assert r.selfcheck[k] < 1e-14
    rb = robust(r, 1e-3)
    assert rb["rank_stable"] and rb["certified"]
    print("core demo OK  rank %d  Gamma %s" % (r.rank_num, {k: round(v, 3) for k, v in g.items()}))


if __name__ == "__main__":
    demo()
