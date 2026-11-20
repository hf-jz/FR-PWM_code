"""A closed loop that can actually move: the physics core and its learned residual are the plant.

The replay plant cannot be actuated, because its temperature profile is a recording; the run in stage57 shows the
command being accepted and changing nothing. The only plant on this machine that responds to a command is the world
model itself. Closing the loop on it answers a narrower question than a device would: with a controller asking for a
core temperature and the same robust barrier screening the command, does the closed loop converge, and how much of
the request survives screening.

Writes stage59/model_loop.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_cbf"))
os.makedirs("stage59", exist_ok=True)
from frpwm.data.mast_replay import MastReplayPlant      # noqa: E402
from frpwm.runtime.pipeline import FrpwmMVP            # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm_p0.cbf import robust_cbf_margin             # noqa: E402

STEPS = 400
TE_REF, TE_MIN, TE_MAX = 400.0, 150.0, 8000.0
ALPHA = 0.3
GAIN = 0.003      # a candidate of order 0.3 for a 100 eV error, which is what the actuator can accept

plant = MastReplayPlant("/tmp/mast28/28315.zarr")
app = FrpwmMVP(plant=plant)
app.train_neural(n=40, verbose=False)
world = app.world
N = len(plant.rho)

X = np.full(N, 500.0)          # start above the reference, so a negative command has work to do
rng = np.random.default_rng(11)


def f_xu(x, u):
    return np.asarray(world.step(np.asarray(x, float), float(np.asarray(u).reshape(-1)[0]))[1], float).reshape(-1)


def h(x):
    te = np.asarray(x, float).reshape(-1)
    return float(min(np.min(te - TE_MIN), np.min(TE_MAX - te)))


lf = 0.0
for _ in range(60):
    x = np.full(N, rng.uniform(200.0, 800.0))
    dx = rng.uniform(1e-2, 1.0)
    lf = max(lf, float(np.linalg.norm(f_xu(x + dx, 0.0) - f_xu(x, 0.0)) / dx))
lf *= 1.2
lh, eps, wbar = 1.0, 0.05 * TE_REF, 6.4

traj, margins, applied, wanted, released = [], [], [], [], 0
for k in range(STEPS):
    core = float(np.mean(X[: max(3, N // 4)]))
    u_c = float(np.clip(GAIN * (TE_REF - core), -1.0, 1.0))
    m = robust_cbf_margin(h, f_xu, X, [u_c], Lh=lh, Lf=lf, eps=eps, wbar=wbar, alpha=ALPHA)
    if m >= 0:
        u, released = u_c, released + 1
    else:
        u = 0.0
    margins.append(float(m))
    wanted.append(u_c)
    applied.append(float(u))
    X = f_xu(X, u)
    if not np.all(np.isfinite(X)):
        break
    traj.append(float(np.mean(X[: max(3, N // 4)])))

traj = np.array(traj)
out = dict(steps=int(traj.size), reference_eV=TE_REF, band_eV=[TE_MIN, TE_MAX],
           released=int(released), released_share=float(released / max(1, traj.size)),
           u_applied_abs_mean=float(np.mean(np.abs(applied))), u_wanted_abs_mean=float(np.mean(np.abs(wanted))),
           core_start_eV=500.0, core_end_eV=float(traj[-1]), core_min_eV=float(traj.min()),
           core_max_eV=float(traj.max()), settling_error_eV=float(traj[-1] - TE_REF),
           overshoot_eV=float(traj.max() - TE_REF), divergent=bool(not np.all(np.isfinite(traj))),
           margin_min=float(np.min(margins)), margin_median=float(np.median(margins)),
           lipshitz=float(lf),
           note=("the plant here is the world model, so this is a model-in-the-loop run: it shows that the closed loop "
                 "settles and how often the barrier passes the command, and it cannot show what a device would do"))

# what would it take for a command to pass at all? sweep the two terms that consume the margin
sweep = []
for eps_rel in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
    Xs, rel = np.full(N, 500.0), 0
    for _ in range(60):
        core = float(np.mean(Xs[: max(3, N // 4)]))
        u_c = float(np.clip(GAIN * (TE_REF - core), -1.0, 1.0))
        m = robust_cbf_margin(h, f_xu, Xs, [u_c], Lh=lh, Lf=lf, eps=eps_rel * TE_REF, wbar=wbar, alpha=ALPHA)
        if m >= 0:
            rel += 1
        else:
            u_c = 0.0
        Xs = f_xu(Xs, u_c)
    sweep.append(dict(eps_rel=eps_rel, released=rel, steps=60))
    print("eps %.3f -> released %d/60" % (eps_rel, rel))
out["eps_sweep"] = sweep
lf_sweep = []
for f in (1.0, 0.5, 0.2, 0.1, 0.05):
    Xs, rel = np.full(N, 500.0), 0
    for _ in range(60):
        core = float(np.mean(Xs[: max(3, N // 4)]))
        u_c = float(np.clip(GAIN * (TE_REF - core), -1.0, 1.0))
        m = robust_cbf_margin(h, f_xu, Xs, [u_c], Lh=lh, Lf=lf * f, eps=eps, wbar=wbar, alpha=ALPHA)
        if m >= 0:
            rel += 1
        else:
            u_c = 0.0
        Xs = f_xu(Xs, u_c)
    lf_sweep.append(dict(lf_scale=f, released=rel, steps=60))
    print("Lf x%.2f -> released %d/60" % (f, rel))
out["lf_sweep"] = lf_sweep

json.dump(out, open("stage59/model_loop.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
