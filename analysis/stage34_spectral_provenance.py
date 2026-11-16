"""Spectral provenance: instrument width, and whether an external code can serve as an independent generator.

The spectral channels reported in the paper are synthetic. This script quantifies the largest omission in that model
and tests whether the archived output of another code could remove it. Three blocks, one record (A16):

1. The per-chord Gaussian width of the real charge-exchange spectra, which the generator does not reproduce: the
   synthetic band keeps a single assumed width with no chord dependence.
2. What a width error does to a frame. The line integral is invariant, so a ratio-based inversion is insensitive to
   it; the pixel shape is not, so the same error is a shape error of tens of per cent.
3. The five properties an independent generator would need, tested against the archived external state.

Writes to $FRPWM_OUT, never over an archived record.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "library"))
from paths import OUT, record  # noqa: E402

FWHM = 0.15                      # Angstrom, the width assumed by the synthetic band
BAND = json.load(open(record("A16_Spectral_Provenance", "line_centers.json")))
WL = np.linspace(BAND["band"][0], BAND["band"][1], BAND["n_pix"])

# 1. the measured width and its chord-to-chord spread
real = json.load(open(record("A5_MAST_60Shot_Replay", "mast28_real_spectra_summary.json")))["per_shot"]
med = float(np.median([v["median"] for v in real.values()]))
p10 = float(np.median([v["p10"] for v in real.values()]))
p90 = float(np.median([v["p90"] for v in real.values()]))
width = dict(median=med, p10=p10, p90=p90, chord_factor=p90 / p10)

# 2. what a mismatch of that size does to the frame
centers = np.array(BAND["centers_ang"], dtype=float)
lines = np.random.default_rng(3).lognormal(0.0, 1.0, size=centers.size)


def frame(fwhm):
    sig = fwhm / 2.355
    return (lines[:, None] / (sig * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((WL[None, :] - centers[:, None]) / sig) ** 2)).sum(0)


base = frame(FWHM)
peak = base > 1e-3 * base.max()
frame_mismatch = {}
for tag, k in (("matched", 1.0), ("narrower_true", width["chord_factor"]), ("wider_true", 1.0 / width["chord_factor"])):
    y = frame(FWHM * k)
    dev = np.abs(y - base) / np.maximum(base, 1e-30)
    frame_mismatch[tag] = dict(width_factor=k,
                               median_pixel_deviation=float(np.median(dev[peak])),
                               p90_pixel_deviation=float(np.percentile(dev[peak], 90)),
                               total_integral_ratio=float(y.sum() / base.sum()))
sig = FWHM / 2.355
G = np.exp(-0.5 * ((WL[None, :] - centers[:, None]) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
frame_mismatch["direct_inverse"] = dict(condition_number=float(np.linalg.cond(G.T)))

# 3. readiness of the archived external state
st = json.load(open(record("A16_Spectral_Provenance", "jorek_state_report.json")))
imp = json.load(open(record("A16_Spectral_Provenance", "jorek_imp_profile.json")))
ver = json.load(open(record("A16_Spectral_Provenance", "jorek_imp_verdict.json")))
checks = dict(
    density_absolute=st["ne_norm_range"][1] > st["ne_norm_range"][0] + 1e-9,
    temperature_absolute=st["Te_norm_range"][1] > 1.0,
    impurity_profile_resolved=bool(ver.get("resolved")),
    impurity_fit_quality=imp["rms_log_residual"] < 0.3,
    multiple_time_slices=False,
)
readiness = dict(checks={k: bool(v) for k, v in checks.items()},
                 usable_as_independent_state=bool(checks["density_absolute"] and checks["temperature_absolute"]),
                 usable_as_independent_diagnostic=False,
                 requirement=("absolute ne and Te in eV on the flux grid, an impurity density resolved below the bin "
                              "scale, and at least one time sequence"))

out = dict(sigma_pix_measured=width, frame_mismatch=frame_mismatch, jorek_readiness=readiness)
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "spectral_provenance.json")
json.dump(out, open(path, "w"), indent=1)
print("measured width: median %.2f px  p10 %.2f  p90 %.2f  chord factor %.2f" % (med, p10, p90, width["chord_factor"]))
for tag in ("narrower_true", "wider_true"):
    print("  %-14s median pixel dev %.3f  integral ratio %.6f"
          % (tag, frame_mismatch[tag]["median_pixel_deviation"], frame_mismatch[tag]["total_integral_ratio"]))
print("external state usable as an independent generator:", readiness["usable_as_independent_state"])
print("wrote", path)
