"""Figure 3 of the PST paper: every number is read from an archived JSON artefact.

Sources
  frpwm/tests/results/upsse/report.json          closure, isotopes, impurity, R^2, coverage
  frpwm/tests/results/pgm_zeff/report.json       effective charge, three paths, two tasks
  frpwm/tests/results/inversion/inversion_report.json   D/V recovery and the pinch cost ratio
  frpwm/tests/results/strahl_pinn/strahl_pinn_report.json  D-dimension cost ratio
No value is typed into this file.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import numpy as np

REPO = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
U = json.loads(Path(record("tests/results/upsse/report.json")).read_text())
Z = json.loads(Path(record("tests/results/pgm_zeff/report.json")).read_text())
I = json.loads(Path(record("tests/results/inversion/inversion_report.json")).read_text())
S = json.loads(Path(record("tests/results/strahl_pinn/strahl_pinn_report.json")).read_text())

closure = [U["closure_euv_pct"], U["closure_bal_pct"], U["iso"], U["cw"]]
nz_r2, cover = U["nz_r2"], U["picp90"]
clean = [Z["exp1"]["baseline_A"], Z["exp1"]["pgm_physics"], Z["exp1"]["pgm_fused"]]
degen = [Z["exp2"]["baseline_A"], Z["exp2"]["pgm_physics"], Z["exp2"]["pgm_fused"]]
factor = degen[0] / degen[2]
dv_grid = I["dv"]["case_B"]["err"]
dv_in = I["dv"]["case_A"]["err"]
ratio_pinch = float(I["checks"][2]["detail"].split("最优 ")[1].split("×")[0])
ratio_d = float(S["checks"][1]["detail"].split("最优 ")[1].split("×")[0])

BLUE, OLIVE, GRAY, RED, GREEN = "#4A7FB5", "#B3801F", "#B8B8B8", "#B85454", "#3C8266"
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.0,
                     "xtick.major.width": 1.0, "ytick.major.width": 1.0})


def bars(ax, labels, vals, colors, ylim, ylabel, title, rotate=90, fmt="{:.2f}%"):
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ylim[1] * 0.015, fmt.format(v), ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", rotation=rotate)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)


fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.79), dpi=220)
fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.075, wspace=0.30, hspace=0.42)

ax = axs[0, 0]
bars(ax, ["EUV\nlines", "Balmer\nlines", "Isotopes", "Impurities"], closure,
     [GREEN, BLUE, OLIVE, RED], (0, 12), "Closure error [%]",
     f"(a) Forward-inverse closure\n$n_z\\ R^2$ = {nz_r2:.3f}, coverage = {cover:.3f}", rotate=0)

ax = axs[0, 1]
groups = ["single W\n(clean)", "C + W\n(degenerate)"]
series = [("spectral-only", GRAY, clean[0], degen[0]),
          ("physics path", BLUE, clean[1], degen[1]),
          ("fused model", GREEN, clean[2], degen[2])]
w = 0.26
x = np.arange(2)
for k, (name, col, v0, v1) in enumerate(series):
    vals = [v0, v1]
    ax.bar(x + (k - 1) * w, vals, width=w, color=col, edgecolor="black",
           linewidth=0.8, label=name)
    for xi, v in zip(x + (k - 1) * w, vals):
        ax.text(xi, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=12, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=13)
ax.set_ylim(0, 22); ax.set_ylabel(r"$Z_{\rm eff}$ error [%]", fontsize=13)
ax.set_title(r"(b) $Z_{\rm eff}$ error, per inference path", fontsize=13.5, loc="left")
ax.grid(axis="y", ls="--", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=12, loc="upper left", frameon=False)
ax.text(1 + w * 0.55, 9.5, f"×{factor:.1f}", ha="center", fontsize=15, color=RED)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=12)

ax = axs[1, 0]
bars(ax, ["pinch vs zero", "$D$ vs next"], [ratio_pinch, ratio_d],
     [BLUE, OLIVE], (0, 17), "cost ratio", "(c) Identifiability margin\n(next-best cost / optimum)",
     rotate=0, fmt="{:.1f}×")

ax = axs[1, 1]
bars(ax, [r"$D$", r"$v$"], [dv_grid[0] * 100, dv_grid[1] * 100], [BLUE, OLIVE],
     (0, 6), "parameter error [%]", "(d) $D/v$ inversion, truth between grid nodes", rotate=0)

out = Path(__file__).with_name("figure3_results.png")
fig.savefig(out, dpi=400, facecolor="white")
fig.savefig(str(out).replace(".png", ".pdf"), facecolor="white")

# self-check: the plotted values are the archived ones
assert abs(closure[0] - 1.97) < 0.01 and abs(closure[1] - 9.55) < 0.01
assert abs(clear := dv_in[0]) < 1e-9
assert abs(factor - 4.59) < 0.02 and ratio_pinch > 10 and ratio_d > 5
print("saved", out)
print("closure", [round(v, 2) for v in closure], "R2", round(nz_r2, 3), "coverage", round(cover, 3))
print("zeff clean", [round(v, 2) for v in clean], "degenerate", [round(v, 2) for v in degen], "factor", round(factor, 2))
print("cost ratios", ratio_pinch, ratio_d, "DV between nodes", [round(v * 100, 2) for v in dv_grid])
