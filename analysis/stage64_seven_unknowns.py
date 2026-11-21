"""The seven unknowns of the carbon system, named, and what the six lines can and cannot see.

The manuscript says six observations and seven unknowns without saying what the seventh is. Carbon has six protons, so
its ionisation balance has seven stages, C I to C VII; the six transitions used here are one per stage from C I to
C VI, and no line of the fully stripped stage is in the set. This script builds the operator the runtime actually
builds, prints its singular values, the right null vector, and the projection of three target functionals onto it, so
that the claim about which functional is identifiable is decided by numbers rather than by wording.

Writes stage64/seven_unknowns.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
os.makedirs("stage64", exist_ok=True)
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.physics.adas import LINES, PecCache, ion_fractions   # noqa: E402

LINES6 = ("C I 834.4", "C II 426.6", "C III 288.4", "C IV 222.3", "C V 33.4", "C VI 182.2")
PC = PecCache()
N_STAGE = 7                      # C I .. C VII


def operator(te, ne, fr):
    A = np.zeros((len(LINES6), N_STAGE))
    for k, name in enumerate(LINES6):
        el, q, wl = LINES[name]
        pec = float(PC.pec_value(el, q, wl, ne, te))
        A[k, q] = pec * ne * float(fr.get(int(q), 0.0))  # emissivity per unit total concentration, stage q
    return A


rows = []
for te in (10.0, 30.0, 60.0):
    for ne in (2e13, 1e14):
        fr = {int(q): float(v) for q, v in ion_fractions("c", ne, te)}
        A = operator(te, ne, fr)
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(s > s[0] * 1e-10))
        N = Vt[rank:].T
        ones = np.ones(N_STAGE)
        obs = np.array([1.0] * 6 + [0.0])              # sum over the six observed stages
        single = np.eye(N_STAGE)[0]                     # one charge state
        proj = lambda c: float(np.linalg.norm(N @ (N.T @ c)) / np.linalg.norm(c)) if N.size else 0.0
        rows.append(dict(te=te, ne=ne, rank=rank, singular_values=[float(x) for x in s[:3]] + [float(s[-1])],
                         null_vector=[float(x) for x in N[:, 0]] if N.size else [],
                         null_fraction_ones=proj(ones), null_fraction_observed_sum=proj(obs),
                         null_fraction_single_state=proj(single),
                         cn=float(s[0] / s[-1])))
        print("Te %5.1f ne %.0e | sv %.3e %.3e %.3e %.3e | rank %d" % (te, ne, s[0], s[1], s[2], s[-1], rank))
        print("     null %s | proj(ones) %.3f | proj(observed sum) %.3f | proj(C I) %.3f"
              % (np.round(N[:, 0], 3) if N.size else "-", rows[-1]["null_fraction_ones"],
                 rows[-1]["null_fraction_observed_sum"], rows[-1]["null_fraction_single_state"]))

# a non-negative solve of the physical operator, to show what NNLS returns along the null direction
te, ne = 30.0, 1e14
fr = {int(q): float(v) for q, v in ion_fractions("c", ne, te)}
A = operator(te, ne, fr)
c_true = np.array([0.2, 0.9, 1.8, 2.4, 1.5, 0.7, 0.1])
y = A @ c_true
c_nnls, _ = nnls(A, y)
out = dict(stages=["C I", "C II", "C III", "C IV", "C V", "C VI", "C VII"], lines=list(LINES6),
           grid=rows,
           nnls_example=dict(true=[float(v) for v in c_true], fit=[float(v) for v in c_nnls],
                             total_true=float(c_true.sum()), total_fit=float(c_nnls.sum()),
                             observed_true=float(c_true[:6].sum()), observed_fit=float(c_nnls[:6].sum())))
json.dump(out, open("stage64/seven_unknowns.json", "w"), indent=1)
print("NNLS total %.3f vs true %.3f | observed-sum %.3f vs %.3f"
      % (c_nnls.sum(), c_true.sum(), c_nnls[:6].sum(), c_true[:6].sum()))
