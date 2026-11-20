"""Cross-check the measured line width against the analysed ion temperature of the same instrument.

Both arrays come from the same real shot: `act/c_ss_raw_spectrum` (the raw charge-exchange spectra) and
`act/c_ss_temperature` (the temperature the diagnostic's own analysis reports for those chords). The width is fitted
here; the temperature is read, not computed. One constant — dispersion times the mass factor — is fitted across all
chords and shots, so the question is whether a single constant explains them. Two controls are reported: the same
statistic with the temperatures permuted, and the width-only variance, so a width that does not move with temperature
cannot masquerade as a Doppler measurement.
"""
import json
import os
import sys

import fsspec
import numpy as np
import zarr
from scipy.optimize import curve_fit
from scipy import stats

BASE = "https://s3.echo.stfc.ac.uk/mast/level1/shots/%s.zarr"
SHOTS = [str(s) for s in range(28315, 28325)]
os.makedirs("stage48", exist_ok=True)
rng = np.random.default_rng(11)


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
        p, _ = curve_fit(gauss, x, y, p0=[a0, mu0, 2.5, c0], maxfev=6000)
    except Exception:  # noqa: BLE001
        return None
    a, mu, sig, c = p
    resid = float(np.std(y - gauss(x, *p)))
    if a <= 0 or resid <= 0 or not (0.3 < abs(sig) < y.size / 2) or a / resid < 4.0:
        return None
    return float(abs(sig))


fs = fsspec.filesystem("http")
rows, per_shot = [], []
for shot in SHOTS:
    try:
        g = zarr.open(fs.get_mapper(BASE % shot), mode="r")
        spec = np.asarray(g["act/c_ss_raw_spectrum"][...], float)
        ti = np.asarray(g["act/c_ss_temperature"][...], float)
    except Exception as exc:  # noqa: BLE001
        per_shot.append(dict(shot=shot, error="%s: %s" % (type(exc).__name__, str(exc)[:70])))
        continue
    n_sl, n_ch, n_pix = spec.shape
    n = 0
    for s in range(n_sl):
        for ch in range(min(n_ch, ti.shape[1])):
            T = ti[s, ch]
            if not np.isfinite(T) or not (20.0 <= T <= 2000.0):
                continue
            sig = fit_width(spec[s, ch, :])
            if sig is None or not (0.5 <= sig <= 6.0):
                continue
            rows.append(dict(shot=shot, slice=s, chord=ch, sigma=sig, Ti=float(T)))
            n += 1
    per_shot.append(dict(shot=shot, slices=n_sl, chords=n_ch, pairs=n,
                         Ti_range=[float(np.nanmin(ti[np.isfinite(ti) & (ti > 20)])),
                                   float(np.nanmax(ti[np.isfinite(ti) & (ti > 20)]))] if np.any(np.isfinite(ti) & (ti > 20)) else None))
    print("%s: %d pairs" % (shot, n), flush=True)

out = dict(pairs=len(rows), per_shot=per_shot)
if len(rows) >= 20:
    w2 = np.array([r["sigma"] ** 2 for r in rows])
    T = np.array([r["Ti"] for r in rows])
    k = float((w2 * T).sum() / (T * T).sum())
    ss_tot = float(((w2 - w2.mean()) ** 2).sum())
    r2 = 1.0 - float(((w2 - k * T) ** 2).sum()) / ss_tot
    slope, intercept = np.polyfit(T, w2, 1)
    r2_free = 1.0 - float(((w2 - (slope * T + intercept)) ** 2).sum()) / ss_tot
    rho, pval = stats.spearmanr(T, w2)
    # physically motivated model: an instrument floor per shot plus a common Doppler slope
    shots = sorted({r["shot"] for r in rows})
    X = np.zeros((len(rows), len(shots) + 1))
    for i, r in enumerate(rows):
        X[i, shots.index(r["shot"])] = 1.0
        X[i, -1] = r["Ti"]
    coef, *_ = np.linalg.lstsq(X, w2, rcond=None)
    pred2 = X @ coef
    r2_floor = 1.0 - float(((w2 - pred2) ** 2).sum()) / ss_tot
    out_floor = dict(slope=float(coef[-1]), floors={s: float(c) for s, c in zip(shots, coef[:-1])},
                     r2=float(r2_floor))

    # control: the same statistic with temperatures permuted, averaged over twenty permutations
    rho_null = []
    for _ in range(20):
        perm = rng.permutation(T)
        rho_null.append(stats.spearmanr(perm, w2)[0])
    out.update(dict(per_shot_floor_model=out_floor,
                    doppler_slope=float(k), r2_one_constant=float(r2), r2_free_line=float(r2_free),
                    spearman=float(rho), spearman_p=float(pval),
                    spearman_permuted_mean=float(np.mean(rho_null)),
                    spearman_permuted_max=float(np.max(np.abs(rho_null))),
                    sigma_range=[float(np.sqrt(w2.min())), float(np.sqrt(w2.max()))],
                    Ti_range=[float(T.min()), float(T.max())]))
    print("pairs %d | sigma^2 = %.3e T | R2 one constant %.3f, free line %.3f | Spearman %.3f (p=%.1e)"
          % (len(rows), k, r2, r2_free, rho, pval))
    print("per-shot floor + common slope: slope %.3e, R2 %.3f" % (out_floor["slope"], out_floor["r2"]))
    print("permuted control: mean %.3f, max |rho| %.3f" % (np.mean(rho_null), np.max(np.abs(rho_null))))

json.dump(out, open("stage48/width_vs_ti.json", "w"), indent=1)
print("wrote stage48/width_vs_ti.json")
