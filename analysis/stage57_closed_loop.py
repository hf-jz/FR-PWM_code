"""Close the loop on a plant model: the shield's output enters the plant, and the next frame is its consequence.

Everywhere else in this paper the recorded discharge proceeds regardless of what the chain says. Here the chain has a
controller, the shield screens the controller's candidate, the surviving command is accepted by the replay plant, and
the next frame follows from it. The plant is still a recorded discharge interpolated in time, not a device; what the
run tests is whether the chain, its shield and its fallback ladder behave when their output has consequences.

Two bands are run, the frozen one and a widened one, so that a refusal cannot be blamed on the band alone.

Writes stage57/closed_loop.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
os.makedirs("stage57", exist_ok=True)
from frpwm.data.mast_replay import MastReplayPlant      # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.runtime.pipeline import FrpwmMVP            # noqa: E402

SHOTS = ("28315", "28316", "28317")
FRAMES = 40
BANDS = {"frozen 150-8000 eV": (150.0, 8000.0), "widened 50-8000 eV": (50.0, 8000.0)}
out = {"frames": FRAMES, "by_band": {}}

for band_name, (te_min_band, te_max_band) in BANDS.items():
    for shot in SHOTS:
        zp = "/tmp/mast28/%s.zarr" % shot
        rec = dict(band=band_name, shot=shot)
        if not os.path.isdir(zp):
            rec["error"] = "no replay cache"
            out["by_band"].setdefault(band_name, {})[shot] = rec
            continue
        closed = MastReplayPlant(zp)
        openp = MastReplayPlant(zp)
        app = FrpwmMVP(plant=closed)
        app.control_enabled = True      # without a controller the shield has nothing to screen
        app.train_neural(n=40, verbose=False)

        u_safe, u_cand, te_c, te_o, te_lo, te_hi, status = [], [], [], [], [], [], {}
        for _ in range(FRAMES):
            s = app.step()
            status[s["decision"]["status"]] = status.get(s["decision"]["status"], 0) + 1
            ctrl = s.get("control") or {}
            u = float(ctrl.get("u_safe", 0.0) or 0.0)
            u_safe.append(u)
            u_cand.append(float(ctrl.get("u_cand", 0.0) or 0.0))
            closed.accept(u)
            te = np.asarray(s["profiles"]["Te"], float)
            te_lo.append(float(te.min()))
            te_hi.append(float(te.max()))
            te_c.append(float(te.mean()))
            te_o.append(float(np.mean(np.asarray(openp.frame()["Te"], float))))

        lo, hi = np.array(te_lo), np.array(te_hi)
        rec.update(status=status,
                   released=int(np.sum(np.abs(u_safe) > 1e-9)),
                   u_safe_abs_mean=float(np.mean(np.abs(u_safe))),
                   u_cand_abs_mean=float(np.mean(np.abs(u_cand))),
                   te_min=float(lo.min()), te_max=float(hi.max()),
                   outside_band=int(np.sum((lo < te_min_band) | (hi > te_max_band))),
                   drift_eV=float(te_c[-1] - te_c[0]), drift_open_eV=float(te_o[-1] - te_o[0]),
                   finite=bool(np.all(np.isfinite(te_c))))
        out["by_band"].setdefault(band_name, {})[shot] = rec
        print("%-20s %s: released %2d/40 | u_safe %.3f vs u_cand %.3f | Te %3.0f-%3.0f | outside %d | drift %.1f (open %.1f)"
              % (band_name, shot, rec["released"], rec["u_safe_abs_mean"], rec["u_cand_abs_mean"],
                 rec["te_min"], rec["te_max"], rec["outside_band"], rec["drift_eV"], rec["drift_open_eV"]))

json.dump(out, open("stage57/closed_loop.json", "w"), indent=1)
print("wrote stage57/closed_loop.json")
