"""Controlled bias injection over a horizon: which readout detects a biased coefficient?

The paper's Table 2 used to offer conformal coverage as evidence for the filter's bias term. This
experiment tests that claim instead of asserting it. A plant with a known perturbation of one
coefficient is rolled forward; the filter, keeping its nominal model and its nominal calibration, predicts
recursively from the same initial state. Three readouts are recorded as the perturbation grows: the
interval half-width (fixed at calibration), the coverage that interval attains, and the signed residual
projected on the model's sensitivity directions.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
from frpwm.invert.ensemble import EnsembleWorld          # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.neural.world import PhysicsStep               # noqa: E402

N = 30
DIFF = 0.02
GAIN = 200.0
NOISE = 0.005
HORIZON = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage27_bias")


def rollout(plant, actions, X0, start):
    traj, X = [], X0.copy()
    for k in range(start, min(start + len(actions), len(actions))):
        u = actions[k]
        Xn = plant(X, u)
        traj.append((X.copy(), u, Xn.copy()))
        X = Xn
    return traj


def sensitivity():
    """One-step response of the model to a unit relative change of each coefficient."""
    X = 3000 * (1 - np.linspace(0, 1, N) ** 2) + 50
    base = PhysicsStep(N, diff=DIFF, heat_gain=GAIN)(X, 0.5)
    d_diff = (PhysicsStep(N, diff=DIFF * 1.01, heat_gain=GAIN)(X, 0.5) - base) / 0.01
    d_gain = (PhysicsStep(N, diff=DIFF, heat_gain=GAIN * 1.01)(X, 0.5) - base) / 0.01
    return np.stack([d_diff, d_gain], axis=1)


def main():
    actions = np.random.default_rng(3).uniform(-1, 1, 440)
    X0 = 3000 * (1 - np.linspace(0, 1, N) ** 2) + 50
    nominal = PhysicsStep(N, diff=DIFF, heat_gain=GAIN)
    traj = rollout(nominal, actions, X0, 0)

    rng = np.random.default_rng(11)
    noisy = [(X, u, Y * (1 + NOISE * rng.standard_normal(N))) for X, u, Y in traj]
    ens = EnsembleWorld(N, 5).train(noisy[:320], epochs=60)
    q90 = ens.calibrate(noisy[320:])
    A = sensitivity()

    cases = [("nominal", 1.00, 1.00), ("diff +10%", 1.10, 1.00), ("diff +25%", 1.25, 1.00),
             ("diff +50%", 1.50, 1.00), ("gain +10%", 1.00, 1.10), ("gain +25%", 1.00, 1.25)]

    report = {"q90": q90, "horizon": HORIZON, "noise": NOISE, "cases": {}}
    for name, dscale, gscale in cases:
        plant = PhysicsStep(N, diff=DIFF * dscale, heat_gain=GAIN * gscale)
        seq = rollout(plant, actions, X0, 320)
        Xhat = X0.copy()
        resid_final, hits, steps = None, [], 0
        for k, (_, u, Y) in enumerate(seq[:HORIZON]):
            mu, _ = ens.predict(Xhat, u)
            rel = np.abs(Y - mu) / (np.abs(Y) + 1e-9)
            hits.append(float((rel <= q90).mean()))
            Xhat = mu
            resid_final = Y - mu
            steps = k + 1
        coefs, *_ = np.linalg.lstsq(A * steps, resid_final, rcond=None)   # bias accumulates over the horizon
        report["cases"][name] = {
            "coverage_over_horizon": float(np.mean(hits)),
            "coverage_final_step": hits[-1],
            "half_width": q90,
            "mean_abs_residual_final": float(np.mean(np.abs(resid_final))),
            "mean_signed_residual_final": float(np.mean(resid_final)),
            "recovered_diff": float(coefs[0]), "recovered_gain": float(coefs[1]),
        }
        r = report["cases"][name]
        print("%-10s cov(horizon) %.3f  cov(final) %.2f  |r| %.1f eV  signed %+.1f eV  ddiff %+.3f  dgain %+.3f"
              % (name, r["coverage_over_horizon"], r["coverage_final_step"], r["mean_abs_residual_final"],
                 r["mean_signed_residual_final"], coefs[0], coefs[1]))

    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "bias_injection.json"), "w"), indent=1)
    print("calibrated half-width %.4f, held fixed in every case" % q90)


if __name__ == "__main__":
    main()
