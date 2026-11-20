"""B3 诊断消融图：摘除诊断后 Z_eff 误差与报告不确定度的响应。"""
import json, os
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

D = record("tests/results/ablation")
ORDER = [("none", "none (4)"), ("euv", "EUV (3)"), ("bal", "Balmer (3)"),
         ("brem", "brems (3)"), ("ph", "physics (3)"),
         ("euv+bal", "EUV+Bal (2)"), ("euv+bal+brem", "EUV+Bal+brem (1)"),
         ("euv+bal+brem+ph", "all four (0)")]
rows = []
for key, label in ORDER:
    p = os.path.join(D, f"{key}.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    rows.append(dict(label=label, n=len(key.split("+")),
                     zeff=r["zeff_fused_kK"], sig=r["sig_iou"],
                     cw=r["cw"], cc=r["cc"], iso=r["iso"], nz=r["nz_r2"]))

# 印刷值断言：图不能偏离归档（审稿要求每个图都有原始数据支撑）
_v = {r["label"]: r for r in rows}
assert abs(_v["none (4)"]["zeff"] - 9.745) < 0.01, _v["none (4)"]["zeff"]
assert abs(_v["brems (3)"]["zeff"] - 21.534) < 0.01, _v["brems (3)"]["zeff"]
assert abs(_v["Balmer (3)"]["iso"] - 45.182) < 0.01
assert abs(_v["EUV (3)"]["nz"] - 0.527) < 0.001
assert abs(_v["all four (0)"]["sig"] - 6.851) < 0.01
assert len(rows) == 8, len(rows)

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.bar(x - 0.2, [r["zeff"] for r in rows], 0.4, color="#3b6ea5", label="effective-charge error")
ax.set_ylabel("effective-charge error (%)")
ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in rows], rotation=25, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax2 = ax.twinx()
ax2.plot(x + 0.2, [r["sig"] for r in rows], "o-", color="#c0392b", linewidth=1.8,
         markersize=6, label="reported uncertainty")
ax2.set_ylabel("reported standard deviation (log scale)")
ax2.set_yscale("log")
ax.set_xlabel("diagnostic groups removed (groups retained)")
for a, b in ((ax, 0), (ax2, 1)):
    pass
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "figure8_ablation.%s" % ext), dpi=300)
print(json.dumps(rows, indent=1))
