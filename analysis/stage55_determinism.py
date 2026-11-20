"""Does the same bundle replay the same frames to the same digits on this machine, twice?"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
os.makedirs("stage55", exist_ok=True)
from frpwm.data.mast_replay import MastReplayPlant     # noqa: E402
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.runtime.pipeline import FrpwmMVP           # noqa: E402


def run(shot, n=20):
    plant = MastReplayPlant("/tmp/mast28/%s.zarr" % shot)
    app = FrpwmMVP(plant=plant)
    app.train_neural(n=40)
    h = hashlib.sha256()
    for t in np.linspace(plant.t0, plant.t1, n):
        s = app.step(t=float(t))
        h.update(np.asarray(s["profiles"]["Te"], float).tobytes())
        h.update(str(s["decision"]["status"]).encode())
    return h.hexdigest()[:32]


out = {"runs": {}}
for shot in ("28315", "28317"):
    a, b = run(shot), run(shot)
    out["runs"][shot] = dict(first=a, second=b, identical=a == b)
    print(shot, a, b, "identical" if a == b else "DIFFER")
out["claim"] = ("same bundle, same frames, two independent processes on this machine and this software stack; "
                "cross-machine bitwise identity is not tested and not claimed")
json.dump(out, open("stage55/determinism.json", "w"), indent=1)
