"""Robust discrete-time CBF screening with explicit fallback certification."""
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import numpy as np


@dataclass
class ActionDecision:
    action: Optional[np.ndarray]
    certified: bool
    source: str
    margins: dict
    reason: str


def robust_cbf_margin(h, f_xu, xhat, u, *, Lh, Lf, eps, wbar, alpha):
    """
    Conservative robust discrete-time CBF sufficient-condition margin:
      h(f(xhat,u)) - Lh*(Lf*eps+wbar)
      - (1-alpha)*(h(xhat)+Lh*eps).

    Non-negative margin means the supplied Lipschitz/error-bound condition
    is satisfied. It is conditional on validity of all supplied bounds.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    if min(Lh, Lf, eps, wbar) < 0:
        raise ValueError("Lipschitz/error bounds must be non-negative")
    hx = float(h(np.asarray(xhat, dtype=float)))
    hnext_nom = float(h(np.asarray(f_xu(np.asarray(xhat, dtype=float),
                                       np.asarray(u, dtype=float)), dtype=float)))
    return hnext_nom - Lh * (Lf * eps + wbar) - (1.0-alpha) * (hx + Lh*eps)


def select_certified_action(h, f_xu, xhat, nominal_action,
                            fallback_actions: Iterable, *,
                            Lh, Lf, eps, wbar, alpha, atol=1e-12):
    """
    Try nominal action first, then fallbacks in caller-defined priority order.
    Every candidate is subjected to exactly the same robust CBF test.
    Returns uncertified=None if none pass.
    """
    margins = {}
    candidates = [("nominal", nominal_action)]
    candidates.extend((f"fallback_{i}", a) for i, a in enumerate(fallback_actions))
    for label, action in candidates:
        u = np.asarray(action, dtype=float)
        margin = robust_cbf_margin(h, f_xu, xhat, u, Lh=Lh, Lf=Lf,
                                   eps=eps, wbar=wbar, alpha=alpha)
        margins[label] = float(margin)
        if margin >= -atol:
            return ActionDecision(u, True, label, margins,
                                  "candidate satisfies supplied robust CBF bound")
    return ActionDecision(None, False, "none", margins,
                          "no candidate satisfies robust CBF bound; enter separately proven emergency protocol")
