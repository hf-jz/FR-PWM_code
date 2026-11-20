#!/usr/bin/env python3
"""Figure 7 (uncertainty budget): drawn from the per-frame artefact, not from aggregates.

Source: stage5/data/uncertainty_budget.csv (304 frames of eight MAST discharges; one row per
frame with the five sigmas and the observation residual). The printed numbers are asserted here,
so the figure cannot drift from the data.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = record("A6_Uncertainty_Budget/uncertainty_budget.csv")

rows = list(csv.DictReader(open(SRC)))
assert len(rows) == 304, len(rows)
ratio = np.array([float(r["normalized_residual"]) for r in rows])
src = {k: float(np.mean([float(r["sigma_" + k]) for r in rows]) ** 2)
       for k in ("measurement", "calibration", "epistemic", "model_form", "process")}
tot = sum(src.values())
share = {k: v / tot for k, v in src.items()}

assert abs(np.median(ratio) - 0.246) < 0.002, np.median(ratio)
assert abs(np.percentile(ratio, 90) - 0.30) < 0.005, np.percentile(ratio, 90)
assert abs(ratio.max() - 0.304) < 0.002, ratio.max()
assert abs(share["epistemic"] - 0.48) < 0.01 and abs(share["model_form"] - 0.48) < 0.01
assert abs(share["measurement"] - 0.03) < 0.01 and share["process"] == 0.0
print("ratios: median %.3f p90 %.3f max %.3f" % (np.median(ratio), np.percentile(ratio, 90), ratio.max()))
print("shares:", {k: round(v, 3) for k, v in share.items()})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
a1.hist(ratio, bins=24, color="#3b6ea5", edgecolor="k", linewidth=0.4)
a1.axvline(ratio.max(), color="#c0392b", ls="--", lw=1.2, label=f"observed max {ratio.max():.2f}")
a1.axvline(1.0, color="k", ls="-", lw=1.2, label="declared $1\\sigma$ bound")
a1.set_xlabel(r"$|r_y|/\sigma_{\mathrm{total}}$ per frame")
a1.set_ylabel("frames")
a1.legend(frameon=False, fontsize=8)
a1.set_xlim(0, 1.05)

names = ["meas", "cal", "epi", "mf", "proc"]
vals = [share["measurement"], share["calibration"], share["epistemic"],
        share["model_form"], share["process"]]
a2.bar(names, [100 * v for v in vals], color="#e1691e", edgecolor="k", linewidth=0.5)
for i, v in enumerate(vals):
    a2.text(i, 100 * v + 1.5, f"{100 * v:.1f}", ha="center", fontsize=7.5)
a2.set_ylabel("share of $\\sigma^2$ (%)")
a2.set_ylim(0, 60)
for ax in (a1, a2):
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(HERE, f"figure7_budget.{ext}")
    fig.savefig(out, dpi=600 if ext == "png" else None)
    print("->", out)
