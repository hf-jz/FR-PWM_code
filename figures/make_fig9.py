"""Figure 9: multi-step rollout error against horizon (T3-E4), read from the artefact."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
import numpy as np

D = json.loads(Path(record("tests/results/t3e4_rollout.json")).read_text())
H = np.array(D["horizons"])

fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=240)
ax.plot(H, np.array(D["physics_in"]) * 100, "o-", color="#5F6670", lw=1.8, ms=5, label="physics core, in range")
ax.plot(H, np.array(D["mixed_in"]) * 100, "s-", color="#2F5D8C", lw=2.0, ms=5, label="core + residual, in range")
ax.plot(H, np.array(D["mixed_out"]) * 100, "^-", color="#B85454", lw=1.8, ms=5, label="core + residual, out of range")
ax.plot(H, np.array(D["fallback_out"]) * 100, "v--", color="#B3801F", lw=1.6, ms=5, label="fallback to core, out of range")
ax.set_xlabel("rollout horizon $k$")
ax.set_ylabel(r"relative error $E(k)$ [%]")
ax.set_xlim(1, H.max()); ax.set_ylim(0, None)
ax.grid(ls="--", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = Path(__file__).with_name("figure9_rollout.png")
fig.savefig(out, facecolor="white")
fig.savefig(str(out).replace(".png", ".pdf"), facecolor="white")
assert abs(D["mixed_in"][-1] / D["physics_in"][-1] - 0.16) < 0.02
print("saved", out, "ratio at k=20:", round(D["physics_in"][-1] / D["mixed_in"][-1], 2))
