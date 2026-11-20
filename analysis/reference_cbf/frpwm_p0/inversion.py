"""Identifiability-gated non-negative multi-line inverse problem."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class InversionResult:
    x: np.ndarray
    singular_values: np.ndarray
    effective_rank: int
    condition_number: float
    identifiable: bool
    weighted_residual_norm: float
    regularized_objective: float
    message: str


def solve_multiline(y, H, sigma, *, R=None, x_prior=None, lambda_reg=0.0,
                    min_rank=None, max_condition=1e8, rank_rtol=1e-10):
    """
    Solve min_{x>=0} ||W(Hx-y)||² + lambda ||R(x-x_prior)||².

    Identifiability is evaluated on the *data* Jacobian W H, not on the
    augmented regularized matrix. Regularization must not manufacture
    observational identifiability.
    """
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if y.ndim != 1 or H.ndim != 2 or H.shape[0] != y.size:
        raise ValueError("Require y:(m,), H:(m,n), with H.shape[0] == len(y)")
    if sigma.shape != y.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite, positive, and match y")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(H)):
        raise ValueError("y and H must be finite")
    m, n = H.shape
    if min_rank is None:
        min_rank = n
    if not (0 <= min_rank <= min(m, n)):
        raise ValueError("min_rank outside valid range")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative")

    Hw = H / sigma[:, None]
    yw = y / sigma
    s = np.linalg.svd(Hw, compute_uv=False)
    tol = (s[0] * rank_rtol) if s.size and s[0] > 0 else 0.0
    rank = int(np.sum(s > tol))
    positive = s[s > tol]
    cond = float(positive[0] / positive[-1]) if positive.size else float("inf")
    identifiable = rank >= min_rank and cond <= max_condition

    if R is None:
        R = np.eye(n)
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[1] != n or np.any(~np.isfinite(R)):
        raise ValueError("R must be finite 2-D with n columns")
    if x_prior is None:
        x_prior = np.zeros(n)
    x_prior = np.asarray(x_prior, dtype=float)
    if x_prior.shape != (n,) or np.any(~np.isfinite(x_prior)):
        raise ValueError("x_prior must be finite with shape (n,)")

    A = Hw
    b = yw
    if lambda_reg > 0:
        scale = np.sqrt(lambda_reg)
        A = np.vstack([A, scale * R])
        b = np.concatenate([b, scale * (R @ x_prior)])

    fit = lsq_linear(A, b, bounds=(0.0, np.inf), method="trf")
    x = fit.x
    residual = Hw @ x - yw
    reg_residual = R @ (x - x_prior)
    data_norm = float(np.linalg.norm(residual))
    objective = float(data_norm**2 + lambda_reg * np.linalg.norm(reg_residual)**2)

    msg = "identifiable under supplied thresholds" if identifiable else (
        "NON-IDENTIFIABLE: report only identifiable combinations or prior-dominated estimates"
    )
    return InversionResult(x, s, rank, cond, identifiable, data_norm, objective, msg)


def charge_state_rhs(ne, S, alpha, n):
    """
    Local adjacent-charge-state kinetics:
      dn_z/dt = ne * [S_{z-1} n_{z-1} - (S_z+alpha_z)n_z
                      + alpha_{z+1}n_{z+1}]
    Boundary rates are interpreted as zero. This function is a local
    reaction-only term; transport/source terms must be added by the caller.
    """
    n = np.asarray(n, dtype=float)
    S = np.asarray(S, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if n.ndim != 1 or S.shape != n.shape or alpha.shape != n.shape:
        raise ValueError("n, S, alpha must be same-length vectors")
    if ne < 0 or np.any(n < 0) or np.any(S < 0) or np.any(alpha < 0):
        raise ValueError("densities and rates must be non-negative")
    out = np.zeros_like(n)
    for z in range(n.size):
        ion_in = S[z-1] * n[z-1] if z > 0 else 0.0
        ion_out = S[z] * n[z]
        rec_in = alpha[z+1] * n[z+1] if z+1 < n.size else 0.0
        rec_out = alpha[z] * n[z]
        out[z] = ne * (ion_in - ion_out - rec_out + rec_in)
    return out
