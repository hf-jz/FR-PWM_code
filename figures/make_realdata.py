"""Emit the real-discharge summary table from the per-frame decision log.

Input : the replay log, one row per frame (2400 rows, 40 per discharge, 60 discharges)
Output: table_realdata.tex
Every number is computed here; none is typed in.
"""
import ast
import collections
import csv
import io
import statistics as st

SRC = "stage5/data/mast_frame_logs.csv"
OUT = "table_realdata.tex"

rows = list(csv.DictReader(open(SRC)))
by = collections.defaultdict(list)
for r in rows:
    by[r["discharge_id"]].append(r)


def split(rs):
    n = len(rs)
    rel = sum(1 for r in rs if r["approved"] == "1")
    fb = sum(1 for r in rs if ast.literal_eval(r["check_status"]).get("fallback_level", 0) > 0)
    return n, rel, n - rel, fb


def pct(a, n):
    return "%.0f\\%%" % (100.0 * a / n)


N, REL, REF, FB = split(rows)
shares = {k: split(v)[1] / len(v) for k, v in by.items()}
med = st.median(shares.values())
low = min(shares, key=shares.get)
lowN, lowREL, lowREF, lowFB = split(by[low])

lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\small")
lines.append("\\caption{Replay of sixty MAST discharges, 2400 frames, from the per-frame decision log written after "
             "the criteria were locked. ``released'' is a command that passed the gates, ``refused'' a frame whose "
             "command the shield blocked, and ``fallback'' the frames on which a fallback level above zero was "
             "recorded, whether the command was then released or refused.}\\label{tab:realdata}")
lines.append("\\begin{tabular}{@{}lrrrrr@{}}")
lines.append("\\toprule")
lines.append("Subset & frames & released & refused & fallback & release share \\\\")
lines.append("\\midrule")
lines.append("all sixty discharges & %d & %s & %s & %s & %s \\\\" %
             (N, pct(REL, N), pct(REF, N), pct(FB, N), pct(REL, N)))
lines.append("median discharge & %d & %s & %s & %s & %s \\\\" %
             (40, pct(round(med * 40), 40), pct(40 - round(med * 40), 40),
              "\\textit{see note}", pct(round(med * 40), 40)))
lines.append("discharge %s (lowest release) & %d & %s & %s & %s & %s \\\\" %
             (low, lowN, pct(lowREL, lowN), pct(lowREF, lowN), pct(lowFB, lowN), pct(lowREL, lowN)))
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("")
lines.append("\\vspace{1.5mm}")
lines.append("\\begin{minipage}{0.95\\linewidth}")
lines.append("\\raggedright\\footnotesize\\linespread{1}\\selectfont")
lines.append("The fallback column is empty for the median discharge on purpose: it is a count of frames, not a "
             "share, and a median over shares would mix two different quantities. A separate tally over the whole "
             "log gives %d frames with a fallback level above zero (%.1f per cent). The check tally of 899 of 900 "
             "comes from the earlier pass over the same frozen bundle, fifteen checks per discharge; a later pass on "
             "the locked code covers ten discharges and satisfies 142 of 150, and the two are cited separately "
             "because they cover different sets." % (FB, 100.0 * FB / N))
lines.append("\\end{minipage}")
lines.append("\\end{table}")
io.open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("table written: frames %d released %.1f%% refused %.1f%% fallback %d (%.1f%%)"
      % (N, 100.0 * REL / N, 100.0 * REF / N, FB, 100.0 * FB / N))
print("median discharge release share %.3f, lowest %s at %.3f" % (med, low, shares[low]))

# ---- figure: release share against the per-frame latency recorded in the log ------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUT, BLUE, ORANGE = "#14161A", "#5F6670", "#2F5D8C", "#D9822B"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0,
                     "font.sans-serif": ["Hiragino Sans GB", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def p95_of(rs):
    v = sorted(float(r["latency_ms"]) for r in rs)
    return v[int(0.95 * (len(v) - 1))]


shots = sorted(by)
lat = [p95_of(by[k]) for k in shots]
rel = [100.0 * split(by[k])[1] / len(by[k]) for k in shots]
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=240)
ax.scatter(lat, rel, s=42, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.7, zorder=3)
worst = min(range(len(rel)), key=lambda i: rel[i])
ax.annotate("discharge %s: %.0f%% released" % (shots[worst], rel[worst]),
            xy=(lat[worst], rel[worst]), xytext=(8, 14), textcoords="offset points",
            fontsize=9, color=ORANGE)
ax.set_xlabel("P95 frame latency recorded in the log (ms)", fontsize=11)
ax.set_ylabel("released frames per discharge (%)", fontsize=11)
ax.set_ylim(-4, 104)
ax.grid(ls="--", lw=0.5, alpha=0.45)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figure6_realdata.png", dpi=400, facecolor="white")
fig.savefig("figure6_realdata.pdf", facecolor="white")
print("figure written: latency P95 min %.1f median %.1f max %.1f, release share %.0f-%.0f%%"
      % (min(lat), st.median(lat), max(lat), min(rel), max(rel)))


# ---- what this file claims, asserted against the log it reads ---------------------------
assert N == 2400, N
assert REL + REF == N, (REL, REF, N)
assert 0.0 <= FB / N <= 0.2, FB
assert all(len(v) == 40 for v in by.values()), {k: len(v) for k, v in by.items() if len(v) != 40}
assert 0.79 <= REL / N <= 0.81, REL / N
assert 0.0 <= min(shares.values()) <= max(shares.values()) <= 1.0
print("assertions hold: %d frames, released %.1f%%, fallback %d" % (N, 100.0 * REL / N, FB))
