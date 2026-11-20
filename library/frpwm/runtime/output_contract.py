"""Target-level output contract: decide, per quantity, whether it may be reported.

The paper's governing principle is that a quantity is emitted only when the current observation set and the physical
model support it. This module implements that decision for a linearised observation operator, so the contract can be
tested rather than asserted. Four classes are returned:

    supported    the target is identifiable, the noise is below its gate, the state is fresh and the model valid
    degraded     identifiable but the interval is wide compared with the tolerance
    withheld     the target lies in or near the null space of the operator, so no value is reported
    invalid      the state is stale beyond its age limit, or the model is outside its validity region

The null-space test is the one that matters: with n observations and m unknowns, a rank below min(n, m) leaves a
direction that no observation constrains, and any quantity with a component along that direction is withheld rather
than returned with a wide interval. Requesting a minimum rank larger than the number of observations is not an error
in the caller; it simply means the rank condition cannot be met, and the state is reported as non-identifiable.
"""
import numpy as np

SUPPORTED, DEGRADED, WITHHELD, INVALID = "supported", "degraded", "withheld", "invalid"


class ContractState:
    """Result of one contract decision, carrying the class, the reason code and the interval."""

    __slots__ = ("cls", "reason", "value", "interval", "null_projection", "rank")

    def __init__(self, cls, reason, value=None, interval=None, null_projection=None, rank=None):
        self.cls, self.reason = cls, reason
        self.value, self.interval = value, interval
        self.null_projection, self.rank = null_projection, rank

    def as_dict(self):
        return dict(cls=self.cls, reason=self.reason, value=self.value, interval=self.interval,
                    null_projection=self.null_projection, rank=self.rank)

    def __repr__(self):
        return "ContractState(%s, %s, value=%s)" % (self.cls, self.reason, self.value)


def _null_basis(A, tol=1e-10):
    """Right null space of A, from the SVD; returns (basis with rows as directions, rank)."""
    u, s, vt = np.linalg.svd(A)
    rank = int((s > tol * max(s.max() if s.size else 1.0, 1e-30)).sum())
    return vt[rank:], rank


def decide(A, y, sigma, target, *, age_ms=0.0, max_age_ms=20.0, model_valid=True,
           tolerance=None, min_rank=None):
    """Decide the output class of one target.

    A       observation matrix, n_obs x n_unknowns
    y       observation vector, n_obs
    sigma   per-observation standard deviation, scalar or length n_obs
    target  weight vector over the unknowns, e.g. ones for the total concentration
    tolerance  optional bound on the half-width; above it the class is degraded
    min_rank   optional rank the caller requires; if it exceeds the number of observations the
                condition cannot be met and the target is withheld as non-identifiable
    """
    A = np.atleast_2d(np.asarray(A, float))
    y = np.atleast_1d(np.asarray(y, float))
    target = np.atleast_1d(np.asarray(target, float))
    n_obs, n_unk = A.shape
    sig = np.full(n_obs, float(sigma)) if np.isscalar(sigma) else np.atleast_1d(np.asarray(sigma, float))
    W = 1.0 / sig
    Aw = A * W[:, None]
    yw = y * W

    if not model_valid:
        return ContractState(INVALID, "model_invalid")
    if age_ms > max_age_ms:
        return ContractState(INVALID, "stale_state")

    basis, rank = _null_basis(Aw)
    if min_rank is not None and min_rank > n_obs:
        # the requested rank is unreachable with this many observations: a condition, not a crash
        return ContractState(WITHHELD, "rank_condition_unreachable", null_projection=1.0, rank=rank)
    if rank < n_unk:
        # the operator leaves a right null space, whatever the number of observations; a target with any
        # component along it cannot be measured, and a rank equal to n_obs does not remove the null space
        # when there are more unknowns than observations
        proj = 0.0
        if basis.size:
            tn = target / max(np.linalg.norm(target), 1e-30)
            proj = float(np.linalg.norm(basis @ tn))
        if proj > 1e-8:
            return ContractState(WITHHELD, "non_identifiable", null_projection=proj, rank=rank)

    # identifiable along the target direction: least squares and the propagated interval
    xn, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    value = float(target @ xn)
    cov = np.linalg.pinv(Aw.T @ Aw)
    half = float(1.645 * np.sqrt(max(target @ cov @ target, 0.0)))
    if tolerance is not None and half > tolerance:
        return ContractState(DEGRADED, "interval_wide", value=value, interval=half, rank=rank)
    return ContractState(SUPPORTED, "ok", value=value, interval=half, rank=rank)


if __name__ == "__main__":
    # six observations, seven unknowns, one transition per charge state: total identifiable, states not
    rng = np.random.default_rng(0)
    # columns 1 and 2 coincide, so the null direction is e1 - e2: the ones vector is orthogonal to it and the
    # total is identifiable, while neither of the two collinear components is
    B = rng.normal(size=(6, 6)) + 3.0 * np.eye(6)
    A = np.column_stack([B[:, 0], B[:, 0], B[:, 1:]])
    assert np.linalg.matrix_rank(A) == 6
    truth = np.array([0.2, 1.0, 3.0, 5.0, 4.0, 2.0, 0.8])
    y = A @ truth
    total = np.ones(7)

    s = decide(A, y, 0.01, total)
    assert s.cls == SUPPORTED and abs(s.value - truth.sum()) < 1e-8, s
    s_state = decide(A, y, 0.01, np.eye(7)[0])   # a component of the collinear pair, in the null space
    assert s_state.cls == WITHHELD and s_state.reason == "non_identifiable", s_state
    s_missing = decide(A[:5], y[:5], 0.01, total)
    assert s_missing.cls == WITHHELD, s_missing
    s_noisy = decide(A, y, 0.5, total, tolerance=1.0)
    assert s_noisy.cls == DEGRADED, s_noisy
    s_stale = decide(A, y, 0.01, total, age_ms=25.0, max_age_ms=20.0)
    assert s_stale.cls == INVALID and s_stale.reason == "stale_state", s_stale
    s_bad = decide(A, y, 0.01, total, model_valid=False)
    assert s_bad.cls == INVALID and s_bad.reason == "model_invalid", s_bad
    s_rank = decide(A, y, 0.01, total, min_rank=8)
    assert s_rank.cls == WITHHELD and s_rank.reason == "rank_condition_unreachable", s_rank
    print("output contract OK: supported, non-identifiable, missing-observation, degraded, stale, model-invalid,"
          " unreachable-rank")
