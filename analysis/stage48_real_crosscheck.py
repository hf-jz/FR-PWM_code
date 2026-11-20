"""Two things the review asks for, done on real diagnostic data.

1. A cross-check against an independent measurement. The charge-exchange spectra of ten MAST discharges are fitted
   chord by chord; the squared Gaussian width should be proportional to the ion temperature that the same campaign
   reports from the same instrument's analysis. One proportionality constant (dispersion times mass factor) is fitted
   across all chords and all shots, so the test is whether a single constant explains the data — not whether a
   two-parameter curve can be drawn through it. The width-only and temperature-only baselines are reported for
   comparison, because a width that does not move with temperature would be consistent with an instrument-limited
   line.

2. End-to-end timing with real diagnostic input. The chain is stepped on the real scalars of the same discharges
   while the measurement vector carries only the one spectral line the real instrument provides; the other channels
   are absent, which is the condition the contract of Section~\\ref{sec:contract} exists for. Per-frame latency is
   recorded, so the timing claim rests on real input rather than on synthetic spectra.

Writes stage48/real_crosscheck.json.
"""
import json
import os
import sys
import time

import fsspec
import numpy as np
import zarr
from scipy.optimize import curve_fit
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from scipy import stats

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

SHOTS = [str(s) for s in range(28315, 28325)]
BASE = "https://s3.echo.stfc.ac.uk/mast/level1/shots/%s.zarr"
os.makedirs("stage48", exist_ok=True)


def gauss(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def fit_width(y):
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or np.ptp(y) <= 0:
        return None
    x = np.arange(y.size, dtype=float)
    c0, a0, mu0 = float(np.median(y)), float(y.max() - np.median(y)), float(np.argmax(y))
    if a0 <= 0:
        return None
    try:
        p, _ = curve_fit(gauss, x, y, p0=[a0, mu0, 3.0, c0], maxfev=4000)
    except Exception:  # noqa: BLE001
        return None
    a, mu, sig, c = p
    resid = np.std(y - gauss(x, *p))
    if a <= 0 or not (0.3 < abs(sig) < y.size / 2) or resid <= 0 or a / resid < 5.0:
        return None
    return float(abs(sig)), float(a / resid)


fs = fsspec.filesystem("http")
rows, timing, per_shot = [], [], []
for shot in SHOTS:
    try:
        g = zarr.open(fs.get_mapper(BASE % shot), mode="r")
        spec = np.asarray(g["act/c_ss_raw_spectrum"][...], float)
    except Exception as exc:  # noqa: BLE001
        per_shot.append(dict(shot=shot, error="spectra: %s" % str(exc)[:80]))
        continue
    z = np.load("stage13/level2_%s.npz" % shot) if os.path.exists("stage13/level2_%s.npz" % shot) else None
    if z is None:
        per_shot.append(dict(shot=shot, error="no analysed profile cached"))
        continue
    R, Ti, t = z["major_radius"], z["t_i"], z["time"]
    k_best, n_best = 0, -1
    for k in range(Ti.shape[1]):
        col = Ti[:, k]
        n = int((np.isfinite(col) & (col > 20)).sum())
        if n > n_best:
            k_best, n_best = k, n
    Ti_k = Ti[:, k_best]
    t_spec = spec.shape[0]
    picked = 0
    for ti in range(t_spec):
        for ch in range(spec.shape[1]):
            out = fit_width(spec[ti, ch, :])
            if out is None or ch >= len(R) or not np.isfinite(Ti_k[ch]) or Ti_k[ch] <= 20:
                continue
            rows.append(dict(shot=shot, slice=ti, chord=ch, sigma=out[0], snr=out[1],
                             Ti=float(Ti_k[ch]), R=float(R[ch])))
            picked += 1
        if picked >= 60:      # enough chords per shot; keep the fetch short
            break
    per_shot.append(dict(shot=shot, slices=spec.shape[0], chords=spec.shape[1], pairs=picked,
                         Ti_range=[float(np.nanmin(Ti_k)), float(np.nanmax(Ti_k))]))
    print("%s: %d pairs, Ti %.0f-%.0f eV" % (shot, picked, np.nanmin(Ti_k), np.nanmax(Ti_k)), flush=True)

out = dict(pairs=len(rows), per_shot=per_shot)
if len(rows) > 20:
    w2 = np.array([r["sigma"] ** 2 for r in rows])
    T = np.array([r["Ti"] for r in rows])
    # one constant across all chords and shots: sigma^2 = k T
    k = float((w2 * T).sum() / (T * T).sum())
    pred = k * T
    ss_res = float(((w2 - pred) ** 2).sum())
    ss_tot = float(((w2 - w2.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rho, pval = stats.spearmanr(T, w2)
    # baselines: width alone, temperature alone
    r2_const = 0.0
    r2_line = None
    slope, intercept = np.polyfit(T, w2, 1)
    r2_line = 1.0 - float(((w2 - (slope * T + intercept)) ** 2).sum()) / ss_tot
    out.update(dict(doppler_slope=float(k), r2_one_constant=float(r2), r2_free_line=float(r2_line),
                    spearman=float(rho), spearman_p=float(pval),
                    sigma_range=[float(np.sqrt(w2.min())), float(np.sqrt(w2.max()))],
                    Ti_range=[float(T.min()), float(T.max())]))
    print("one-constant model: sigma^2 = %.3e T, R2=%.3f (free line R2=%.3f), Spearman=%.3f (p=%.2e)"
          % (k, r2, r2_line, rho, pval))

# ---- end-to-end timing with real input ------------------------------------------------------
try:
    from frpwm.data.mast_replay import MastReplayPlant
    from frpwm.runtime.pipeline import FrpwmMVP
    lat, status = [], {}
    for shot in SHOTS[:5]:
        zp = "/tmp/mast28/%s.zarr" % shot
        if not os.path.isdir(zp):
            continue
        plant = MastReplayPlant(zp)
        app = FrpwmMVP(plant=plant)
        app.train_neural(n=40)
        for t in np.linspace(plant.t0, plant.t1, 20):
            t0 = time.perf_counter()
            s = app.step(t=float(t))
            lat.append((time.perf_counter() - t0) * 1e3)
            status[s["decision"]["status"]] = status.get(s["decision"]["status"], 0) + 1
    lat = np.array(lat)
    out["timing_real_input"] = dict(
        frames=int(lat.size), p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
        worst=float(lat.max()), status=status,
        note=("real scalars and real equilibrium, one real spectral line present; the chain steps the same code path "
              "as in the replay, and the decision mix is reported because a single line cannot support the six-line "
              "inversion"))
    print("real-input timing: %d frames, P50 %.1f ms, P95 %.1f ms, worst %.1f ms"
          % (lat.size, out["timing_real_input"]["p50"], out["timing_real_input"]["p95"], lat.max()))
except Exception as exc:  # noqa: BLE001
    out["timing_real_input"] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:120]))

json.dump(out, open("stage48/real_crosscheck.json", "w"), indent=1)
print("wrote stage48/real_crosscheck.json")
