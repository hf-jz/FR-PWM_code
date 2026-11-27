"""Admissibility of the robust one-step barrier, from the documented constants alone.

Why this file exists: the previous draft quoted a sweep ("commands begin to pass at 0.5 per cent
uncertainty, or at a tenth of the measured Lipschitz constant") whose generating code contained an
empty loop and whose archived record holds one frame repeated 120 times.  Neither number could be
reproduced.  This recomputes what the inequality actually says, so every number in the paper is a
function of constants a reader can check.

Inequality (supplementary material, the shield as a viability condition):

    h(f(x,u)) - L_h (L_f eps + wbar) - (1 - a)(h(x) + L_h eps) >= 0

With the nominal barrier h(f) >= (1-a) h this holds whenever

    h >= Delta/(1-a),      Delta = L_h (L_f eps + wbar) + (1-a) L_h eps                (*)

so the invariant set is the sublevel set of (*), not the nominal margin.  Solving (*) for each
constant in turn gives the three breakpoints below.

    python3 analysis/p0_admissibility.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from paths import ARTEFACTS  # noqa: E402

# documented constants: alpha and eps from the replay configuration, L_f and wbar from the
# certificate record of the 120-frame replay, the margin from the safety specification
ALPHA = 0.30
EPS = 20.0          # eV, the documented five per cent of a 400 eV reference
LF = 13.148607574247645
WBAR = 6.384468017867107
LH = 1.0
H_LIM = 350.0       # eV, the nominal margin the design asks for


def delta(a=ALPHA, eps=EPS, lf=LF, wbar=WBAR, lh=LH):
    return lh * (lf * eps + wbar) + (1 - a) * lh * eps


def main():
    d0 = delta()
    out = {}  # type: dict
    out["constants"] = dict(alpha=ALPHA, eps_eV=EPS, Lf=LF, wbar_eV=WBAR, Lh=LH, h_lim_eV=H_LIM)
    out["delta_eV"] = d0
    out["invariant_set_requires_h_eV"] = d0 / (1 - ALPHA)
    out["invariant_set_nonempty"] = bool(d0 / (1 - ALPHA) < 3925.0)   # max of min(Te-150, 8000-Te)
    out["nominal_margin_supported"] = bool(d0 / (1 - ALPHA) <= H_LIM)
    # breakpoints: hold the other three constants and solve for this one
    out["eps_star_eV"] = (H_LIM * (1 - ALPHA) - WBAR) / (LF + 1 - ALPHA)
    out["eps_star_percent_of_400"] = 100.0 * out["eps_star_eV"] / 400.0
    out["eps_reduction_factor"] = EPS / out["eps_star_eV"]
    out["Lf_star"] = (H_LIM * (1 - ALPHA) - (1 - ALPHA) * EPS - WBAR) / EPS
    out["Lf_star_over_Lf"] = out["Lf_star"] / LF
    # largest contraction gain that the inequality admits: Delta = (1-a) h_lim, with Delta linear in (1-a)
    out["alpha_star"] = 1.0 - (LF * EPS + WBAR) / (H_LIM - EPS)
    # the two terms that consume the margin, as the paper quotes them
    out["terms"] = dict(error_term_eV=LH * LF * EPS, nominal_barrier_eV=(1 - ALPHA) * H_LIM,
                        process_term_eV=LH * WBAR, slope_term_eV=(1 - ALPHA) * LH * EPS)
    # a sanity check on the algebra: the stated eps_star must make Delta exactly (1-a) h_lim
    assert abs(delta(eps=out["eps_star_eV"]) - (1 - ALPHA) * H_LIM) < 1e-9
    assert abs(delta(lf=out["Lf_star"]) - (1 - ALPHA) * H_LIM) < 1e-9
    assert abs(delta(a=out["alpha_star"]) - (1 - out["alpha_star"]) * H_LIM) < 1e-9, "alpha breakpoint"
    assert out["alpha_star"] < ALPHA, "the shipped gain must be the one that fails"
    assert out["Lf_star_over_Lf"] < 1 and out["eps_reduction_factor"] > 1
    dest = os.path.join(ARTEFACTS, "A23_Barrier_Admissibility")
    os.makedirs(dest, exist_ok=True)
    json.dump(out, open(os.path.join(dest, "admissibility.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("\n-> %s" % dest)


if __name__ == "__main__":
    main()
