"""Which functional of the seven-stage carbon density is identifiable, per operating point.

Carbon has seven ionisation stages, C I to C VII. The six transitions used here are one per stage from C I to C VI, so
the fully stripped stage has no line. Building the operator the runtime builds — photon emissivity times density times
the ionisation fraction — shows that its rank is not six at any physical temperature, because a stage that carries no
population contributes no signal: the rank is the number of populated observed stages. The identifiable functional is
therefore not the absolute total but the concentration carried by the stages that are both observed and populated,
and the script prints, per temperature, the rank, the projection of the absolute total onto the null space, and the
projection of that functional, which is zero by construction.

Writes stage65/populated_functional.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
os.makedirs("stage65", exist_ok=True)
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.physics.adas import LINES, PecCache, ion_fractions    # noqa: E402

L6 = ("C I 834.4", "C II 426.6", "C III 288.4", "C IV 222.3", "C V 33.4", "C VI 182.2")
STAGES = ["C I", "C II", "C III", "C IV", "C V", "C VI", "C VII"]
NE = 5e13
PC = PecCache()
TOL = 1e-3          # a stage counts as populated above this fraction


def build(te):
    fr = {int(q): float(v) for q, v in ion_fractions("c", NE, float(te))}
    A = np.zeros((len(L6), 7))
    for k, nm in enumerate(L6):
        el, q, wl = LINES[nm]
        A[k, q] = float(PC.pec_value(el, q, wl, NE, float(te))) * NE * fr[q]
    return A, np.array([fr[q] for q in range(7)])


rows = []
for te in (15, 25, 40, 60, 90, 130, 200, 400):
    A, fr = build(te)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > s[0] * 1e-8))
    N = Vt[rank:].T
    populated = [i for i in range(7) if fr[i] > TOL]
    observed = list(range(6))
    support = [i for i in populated if i in observed]
    c_sup = np.zeros(7)
    c_sup[support] = 1.0
    ones = np.ones(7)
    proj = lambda c: float(np.linalg.norm(N @ (N.T @ c)) / np.linalg.norm(c)) if N.size and np.linalg.norm(c) > 0 else 0.0
    # what a non-negative solve does to the absolute total, with a truth in which only populated stages carry density
    c_true = np.zeros(7)
    c_true[support] = np.linspace(1.0, 1.0 + 0.2 * (len(support) - 1), len(support)) if support else []
    y = A @ c_true
    c_fit, _ = nnls(A, y) if len(support) else (np.zeros(7), 0)
    rows.append(dict(te=te, rank=rank, populated=[STAGES[i] for i in populated], support=[STAGES[i] for i in support],
                     null_dim=int(N.shape[1]), singular_values=[float(x) for x in s],
                     proj_absolute_total=proj(ones), proj_support_functional=proj(c_sup),
                     total_true=float(c_true.sum()), total_nnls=float(c_fit.sum()),
                     support_true=float(c_true[support].sum()), support_nnls=float(c_fit[support].sum())))
    r = rows[-1]
    print("Te %5.0f rank %d null %d | populated %-24s | proj(total) %.3f proj(support) %.3f | support %.2f vs %.2f"
          % (te, rank, N.shape[1] if N.size else 0, ",".join(r["populated"]), r["proj_absolute_total"],
             r["proj_support_functional"], r["support_true"], r["support_nnls"]))

out = dict(stages=STAGES, lines=list(L6), ne=NE, rows=rows,
           note=("rank counts singular values above 1e-8 of the largest; the operator is the one the runtime builds, "
                 "photon emissivity times density times ionisation fraction, so a stage with no population has no "
                 "column and the rank equals the number of populated observed stages"))
json.dump(out, open("stage65/populated_functional.json", "w"), indent=1)
print("wrote stage65/populated_functional.json")
