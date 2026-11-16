"""Fig. 5: contraction of the recursive refinement.
Residual sequence is the archived one from frpwm/tests/results/upsse/report.json
("residual_seq", four iterations of the TRM physics-refinement operator).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUT, BLUE, RED = "#14161A", "#5F6670", "#2F5D8C", "#B85454"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0,
                     "font.sans-serif": ["Hiragino Sans GB", "DejaVu Sans"],
                     "axes.unicode_minus": False})

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import json
_D = json.load(open(record("tests/results/ablation/base_folds.json")))
FOLDS = [np.array(f) for f in _D["residual_folds"]]          # 三折，逐折存档
r = np.array(_D["residual_seq"])                              # 折均值，仅用于 rho 标注
k = np.arange(1, len(r) + 1)
rho = r[1:] / r[:-1]
RHO_F = np.array([f[1:] / f[:-1] for f in FOLDS])

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=240)
for i, f in enumerate(FOLDS):
    ax.plot(k, f * 1e3, "o-", lw=1.6, ms=6, alpha=0.85, zorder=3,
            label="fold %d" % (i + 1))
ax.plot(k, r * 1e3, "s--", color=INK, lw=1.4, ms=5, zorder=4, label="fold mean")
for i, (kk, rr, pp) in enumerate(zip(k[1:], r[1:], rho)):
    lo, hi = RHO_F[:, i].min(), RHO_F[:, i].max()
    ax.annotate(r"$\rho\in[%.3f,%.3f]$" % (lo, hi), xy=(kk, rr * 1e3), xytext=(6, 10),
                textcoords="offset points", fontsize=10, color=MUT)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("refinement iteration $k$", fontsize=12)
ax.set_ylabel(r"normalized observation residual $r^{(k)}$ [$10^{-3}$]", fontsize=12)
ax.set_xticks(k)
ax.set_xlim(0.6, len(r) + 0.9)
ax.grid(ls="--", lw=0.6, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("monotone but slowly contracting: " + r"$\prod \rho_k = %.3f$" % np.prod(rho),
             fontsize=11.5, loc="left", color=MUT, pad=8)
fig.tight_layout()

out = os.path.join(OUT, "figure5_trm.png")
fig.savefig(out, dpi=400, facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), facecolor="white")

assert (r < 0).sum() == 0
assert np.all(rho < 1), rho
assert abs(np.prod(rho) - 0.765) < 0.02, np.prod(rho)
print("saved", out, "| ratios", np.round(rho, 4), "| product", round(float(np.prod(rho)), 4))
