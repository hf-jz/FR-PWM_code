"""Fig. 4: per-term closure error against the pre-registered criteria.
Measured values: frpwm/tests/results/upsse/report.json (closure_euv_pct,
closure_bal_pct, iso, cw). Criteria: the frozen 2.3-gap family
(EUV < 10 %, Balmer < 12 %, isotopes < 12 %, impurity conc. < 10 %).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import numpy as np

INK, MUT, GREEN, RED = "#14161A", "#5F6670", "#2E7D32", "#B85454"
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.0,
                     "font.sans-serif": ["Hiragino Sans GB", "DejaVu Sans"],
                     "axes.unicode_minus": False})

items = ["EUV\nlines", "Balmer\nlines", "isotopes", "impurity\nconc."]
meas = [1.97, 9.55, 8.59, 7.91]
crit = [10.0, 12.0, 12.0, 10.0]

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=220)
x = np.arange(len(items))
ax.bar(x, meas, width=0.54, color=GREEN, edgecolor="black", linewidth=0.7, zorder=3)
for xi, (m, c) in enumerate(zip(meas, crit)):
    ax.hlines(c, xi - 0.33, xi + 0.33, colors=RED, linestyles=(0, (4, 2)),
              linewidth=1.7, zorder=4)
    ax.text(xi, m + 0.55, f"{m:.2f}%", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color=INK, zorder=5)
ax.text(len(items) - 0.42, 12.6, "pre-registered criterion", ha="right", va="bottom",
        fontsize=11.5, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(items, fontsize=13)
ax.set_ylabel("closure error [%]", fontsize=13)
ax.set_ylim(0, 14)
ax.set_xlim(-0.55, len(items) - 0.35)
ax.grid(axis="y", ls="--", lw=0.6, alpha=0.55, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

out = os.path.join(OUT, "fig4_criteria_en.png")
fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=400)
fig.savefig(out.replace('.png', '.pdf'), bbox_inches="tight", facecolor="white")

assert all(m < c for m, c in zip(meas, crit)), list(zip(meas, crit))
assert abs(meas[0] - 1.9712) < 0.01 and abs(meas[1] - 9.5504) < 0.01
print("saved", out, "| bars:", [f"{m}%<{c}%" for m, c in zip(meas, crit)])
