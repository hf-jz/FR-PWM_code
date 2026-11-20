"""A bound on the neural residual that holds over the whole declared set, not over samples.

The protection certificate of Appendix~C needs an upper bound on the residual term that holds for every state in
{max T_e <= 4000 eV}, not on the states that happened to be sampled. The residual is a two-hidden-layer tanh network,
so interval propagation through it gives such a bound: an affine map sends an input box to an output box, and tanh is
monotone and bounded, so the propagated interval contains the network output for every input in the box. The bound is
sound and loose, and the point of computing it is to see whether the empirical allowance survives it.

Writes stage53/certified_bound.json.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library")
sys.path.insert(0, REPO)
os.makedirs("stage53", exist_ok=True)

from frpwm.data.mast_replay import MastReplayPlant          # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.runtime.pipeline import FrpwmMVP                # noqa: E402


def ibp(net, lo, hi):
    """Interval bound propagation through the MLP; returns (lo_out, hi_out)."""
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        c, d = W, b
        mid, rad = (lo + hi) / 2.0, (hi - lo) / 2.0
        zc = mid @ c + d
        zr = np.abs(c).T @ rad
        lo, hi = zc - zr, zc + zr
        if i < len(net.W) - 1:
            lo, hi = np.tanh(lo), np.tanh(hi)
    return lo, hi


plant = MastReplayPlant("/tmp/mast28/28315.zarr")
app = FrpwmMVP(plant=plant)
app.train_neural(n=40)
R = app.world.R
n = app.world.n
mu, sd = app.world._x_mu, app.world._x_sd
ymu, ysd = app.world._y_mu, app.world._y_sd

boxes = {"physical band 150-4000 eV": (150.0, 4000.0), "full non-negative range 0-4000 eV": (0.0, 4000.0)}
out = {"n_nodes": int(n), "hidden": [w.shape[1] for w in R.W[:-1]], "boxes": {}}
for name, (t_lo, t_hi) in boxes.items():
    x_lo = np.concatenate([np.full(n, t_lo), [0.0, 0.0]])
    x_hi = np.concatenate([np.full(n, t_hi), [0.0, 0.0]])
    lo, hi = ibp(R, (x_lo - mu) / sd, (x_hi - mu) / sd)
    lo, hi = lo * ysd + ymu, hi * ysd + ymu
    bound = float(max(abs(lo).max(), abs(hi).max()))
    out["boxes"][name] = dict(residual_lo=float(lo.min()), residual_hi=float(hi.max()), abs_bound_eV=bound,
                             certified_inner_set_eV=4000.0 - bound)
    print("%-34s |R| <= %.1f eV -> inner set %.1f eV" % (name, bound, 4000.0 - bound))

# the empirical allowance the paper quotes, recomputed on the same model for comparison
emp = []
for t in np.linspace(plant.t0, plant.t1, 60):
    s = plant.state_at(float(t)) if hasattr(plant, "state_at") else None
    if s is None:
        continue
    X = np.asarray(s["Te"], float) if isinstance(s, dict) else None
    if X is None:
        continue
    pred = app.world.predict(X, 0.0) if hasattr(app.world, "predict") else None
    if pred is None:
        continue
    resid = np.asarray(pred, float)[0] - app.world.FP(X, 0.0) if np.ndim(pred) > 1 else np.asarray(pred, float) - app.world.FP(X, 0.0)
    emp.append(float(np.abs(resid).max()))
out["empirical_max_eV"] = float(np.max(emp)) if emp else None
out["note"] = ("interval propagation over the input box; sound for every state in the box, and loose by construction. "
               "The empirical maximum is the largest residual seen on the replay, which is what the 5.4 eV allowance "
               "was drawn from.")
json.dump(out, open("stage53/certified_bound.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "boxes"}, indent=1))
