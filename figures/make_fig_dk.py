"""Figure: subspace drift against perturbation size, with the Davis-Kahan bound on the same axes.

Reads the E6 sweep from artefact A22 and draws ||P0' - P0||_2 against ||dJ||_2 in the whitened
operator, together with delta/g.  The point of the figure is that the bound never crosses the
measured drift; the point where the declared rank changes is marked rather than plotted.

    python3 figures/make_fig_dk.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A22 = os.path.join(HERE, "artefacts", "A22_SD_TIF", "benchmarks.json")
OUT = os.path.join(HERE, "generated", "fig_dk_sweep.pdf")


def main():
    b = json.load(open(A22))
    e6 = b["E6 Weyl / Davis-Kahan robustness"]
    rows = [r for r in e6["sweep"] if r["drift"] is not None]
    d = np.array([r["delta"] for r in rows])
    drift = np.array([r["drift"] for r in rows])
    bound = np.array([r["bound"] for r in rows])
    assert np.all(drift <= bound), "the bound must hold everywhere it is plotted"
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.loglog(d, bound, "-", color="#1f3b73", lw=1.4, label=r"Davis-Kahan $\delta_J/g$")
    ax.loglog(d, drift, "o", ms=4, color="#b03a2e", label=r"measured $\|P_0'-P_0\|_2$")
    ax.set_xlabel(r"$\|\Delta\widetilde J\|_2$", fontsize=8, labelpad=2)
    ax.set_ylabel(r"$\|P_0'-P_0\|_2$", fontsize=8, labelpad=6)
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(axis="both", which="both", labelsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title(r"$g=%.0f$, rank declared at $s>%.0f$" % (e6["gap"], e6["tol_num"]), fontsize=7)
    fig.subplots_adjust(left=0.275, right=0.985, top=0.855, bottom=0.235)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
