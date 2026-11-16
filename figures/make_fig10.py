#!/usr/bin/env python3
"""Figure 10 (PST manuscript): (D,V) identifiability under four observation combinations.

Reads tests/results/ident4/ident4_report.json (produced by frpwm/tests/ident_4combos.py) and
asserts every plotted number against it, so the figure cannot drift from the run.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REP = record("tests/results/ident4/ident4_report.json")
rep = json.load(open(REP))
c = rep["combos"]

order = ["single-steady", "multi-steady", "single-time", "multi-time"]
labels = ["one stage\nsteady", "all stages\nsteady", "one stage\ntime-resolved", "all stages\ntime-resolved"]
sD = np.array([c[k]["sigma_D_rel_pct"] for k in order])
sV = np.array([c[k]["sigma_V_rel_pct"] for k in order])

# provenance asserts: the printed bars are the archived values
assert abs(sD[0] - 40.79) < 0.05 and abs(sV[0] - 111.53) < 0.05, (sD[0], sV[0])
assert abs(sD[-1] - 1.55) < 0.05 and abs(sV[-1] - 2.96) < 0.05, (sD[-1], sV[-1])
assert c["multi-time"]["rank"] == 2 and c["single-steady"]["rank"] == 2
assert rep["noise"] == 0.03
print("bars:", sD.round(2), sV.round(2))

fig, ax = plt.subplots(figsize=(5.9, 3.5))
x = np.arange(len(order))
w = 0.36
ax.bar(x - w / 2, sD, w, label=r"$\sigma_D$", color="#4C78A8", edgecolor="k", linewidth=0.5)
ax.bar(x + w / 2, sV, w, label=r"$\sigma_V$", color="#E1691E", edgecolor="k", linewidth=0.5)
for xi, (a, b) in enumerate(zip(sD, sV)):
    for xo, v in ((xi - w / 2, a), (xi + w / 2, b)):
        if v >= 13:
            ax.text(xo, v + 3.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(xo, v * 0.45, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="w", fontweight="bold")
ax.axhline(10.0, color="k", ls="--", lw=0.9, label="required accuracy")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Cramer-Rao lower bound (%)", fontsize=9)
ax.set_ylim(0, 140)
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(HERE, f"figure10_identifiability.{ext}")
    fig.savefig(out, dpi=600 if ext == "png" else None)
    print("->", out)

print(f"closure: sigma_V {sV[1]:.1f}% (all-stage steady) -> {sV[3]:.1f}% (all-stage time-resolved), "
      f"factor {sV[1]/sV[3]:.1f}x; sigma_D factor {sD[1]/sD[3]:.1f}x")
