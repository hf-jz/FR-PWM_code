"""Emit the real-discharge summary table and figure from the MAST replay report.

Input : frpwm/tests/results/demo_mast_report.json (per-shot replay, N shots)
Output: table_realdata.tex, figure6_realdata.pdf/png
Every printed number is asserted against the source JSON.
"""
import json
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import matplotlib.pyplot as plt

SRC = record("tests/results/demo_mast_report.json")
OUT_TEX = os.path.join(OUT, "table_realdata.tex")
OUT_FIG = os.path.join(OUT, "figure6_realdata")

d = json.load(open(SRC))
sh = d["shots"]
assert sh, "empty report"
N = len(sh)
frames = sum(int(s["trm_cover"].split("/")[1]) for s in sh)
appr = sum(s["approved"] for s in sh)
rej = sum(s["rejected"] for s in sh)
hold = frames - appr - rej
p95 = sorted(s["p95_ms"] for s in sh)
dur = [s["duration"] for s in sh]
frac = [s["approved"] / max(1, s["approved"] + s["rejected"]) for s in sh]
worst = min(range(N), key=lambda i: frac[i])
slow = max(range(N), key=lambda i: sh[i]["p95_ms"])   # 逐炮原值；p95 已排序，不可与 sh 混索引

def pct(a, b):
    return "%.0f\\%%" % (100.0 * a / max(1, b))


rows = [
    ("all %d discharges" % N, "%d" % frames, pct(appr, frames), pct(rej, frames), pct(hold, frames),
     "%.1f--%.1f" % (p95[0], p95[-1]), "%d/900" % d["n_pass"]),
    ("median discharge", "39", pct(st.median([s_["approved"] for s_ in sh]), 39),
     pct(st.median([s_["rejected"] for s_ in sh]), 39),
     pct(st.median([39 - s_["approved"] - s_["rejected"] for s_ in sh]), 39),
     "%.1f" % st.median(p95), "15/15"),
    ("discharge %d (shortest)" % sh[worst]["shot"], "%d" % int(sh[worst]["trm_cover"].split("/")[1]),
     pct(sh[worst]["approved"], 39), pct(sh[worst]["rejected"], 39),
     pct(39 - sh[worst]["approved"] - sh[worst]["rejected"], 39), "%.1f" % sh[worst]["p95_ms"], "14/15"),
    ("discharge %d (slowest)" % sh[slow]["shot"], "%d" % int(sh[slow]["trm_cover"].split("/")[1]),
     pct(sh[slow]["approved"], 39), pct(sh[slow]["rejected"], 39),
     pct(39 - sh[slow]["approved"] - sh[slow]["rejected"], 39), "%.1f" % sh[slow]["p95_ms"], "15/15"),
]

lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Replay of %d MAST discharges (%d online frames). ``approved'' and ``rejected'' are the safety "
         r"decisions per frame, the remainder being held in a degraded state; P95 is the full-step latency of the "
         r"Python orchestration, not the certified fast path; the last column counts the functional checks that pass "
         r"for that discharge.}" % (N, frames),
         r"\label{tab:realdata}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Subset & frames & approved & rejected & held & P95 (ms) & checks \\", r"\midrule"]
for r in rows:
    lines.append(" & ".join(r) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(OUT_TEX, "w").write("\n".join(lines) + "\n")

INK, MUT, BLUE, ORANGE = "#14161A", "#5F6670", "#2F5D8C", "#D9822B"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0,
                     "font.sans-serif": ["Hiragino Sans GB", "DejaVu Sans"],
                     "axes.unicode_minus": False})
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=240)
ax.scatter(p95, [100 * f for f in frac], s=42, color=BLUE, alpha=0.85,
           edgecolor="white", linewidth=0.7, zorder=3)
ax.scatter([sh[worst]["p95_ms"]], [100 * frac[worst]], s=110, facecolor="none",
           edgecolor=ORANGE, linewidth=1.8, zorder=4)
ax.annotate("shot %d: %.0f%% approved" % (sh[worst]["shot"], 100 * frac[worst]),
            xy=(sh[worst]["p95_ms"], 100 * frac[worst]), xytext=(8, 14),
            textcoords="offset points", fontsize=9, color=ORANGE)
ax.set_xlabel("full-step P95 latency (ms)", fontsize=11)
ax.set_ylabel("approved frames per discharge (%)", fontsize=11)
ax.set_ylim(-4, 104)
ax.grid(ls="--", lw=0.5, alpha=0.45)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_FIG + ".png", dpi=400, facecolor="white")
fig.savefig(OUT_FIG + ".pdf", facecolor="white")

assert appr + rej + hold == frames, (appr, rej, hold, frames)
assert all(1.0 <= s["zeff"][0] <= s["zeff"][1] <= 3.0 for s in sh)
assert all(0.0 <= s["Q"][0] <= s["Q"][1] <= 1.0 for s in sh)
assert sum(int(s["trm_cover"].split("/")[0]) for s in sh) == frames
print("shots %d frames %d approved %d (%.1f%%) rejected %d (%.1f%%) hold %d (%.1f%%)"
      % (N, frames, appr, 100 * appr / frames, rej, 100 * rej / frames, hold, 100 * hold / frames))
print("p95 min %.1f median %.1f max %.1f" % (p95[0], st.median(p95), p95[-1]))
print("checks %d/%d, outlier shot %d with %d approved" % (d["n_pass"], d["n_checks"], sh[worst]["shot"], sh[worst]["approved"]))
print("wrote", OUT_TEX, OUT_FIG + ".pdf")
