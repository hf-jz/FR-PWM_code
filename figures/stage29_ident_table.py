"""Emit the identifiability table from the A12 artefact.

The section grades targets as I=0/1/2 against a required accuracy and the scale on which the quantity
varies. Neither number was tabulated, so a reader could not reproduce a grade. This reads the archived
Cramer-Rao bounds for the four observation sets and writes them as a LaTeX table, together with the
six-line rank evidence for the fast layer; nothing here is typed by hand.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
import os

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library")
SRC = record("tests/results/ident4/ident4_report.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_ident.tex")

LABEL = {"single-steady": "steady, two lines", "multi-steady": "steady, four lines",
         "single-time": "two lines, eight times", "multi-time": "four lines, eight times"}


def main():
    d = json.load(open(SRC))
    rows = []
    for key in ("single-steady", "multi-steady", "single-time", "multi-time"):
        c = d["combos"][key]
        rows.append((LABEL[key], c["n_obs"], c["rank"], c["cond"],
                     c["sigma_D_rel_pct"], c["sigma_V_rel_pct"]))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Identifiability evidence computed before any model is fitted, read from artefact A12. "
        r"Rank and condition number are those of the weighted Jacobian at the operating point; the two bound "
        r"columns are the marginal Cram\'er--Rao bounds of the identifiability subsection above, as a percentage of the true "
        r"value, with three per cent measurement noise. The last two rows give the fast-layer observation set, "
        r"where the same construction is applied to the six charge states.}",
        r"\label{tab:ident}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"observation set & $n_{\mathrm{obs}}$ & rank & $\kappa$ & $\sigma_D$ (\%) & $\sigma_V$ (\%) \\",
        r"\midrule",
    ]
    for name, n, rank, cond, sd, sv in rows:
        lines.append("%s & %d & %d & %.0f & %.1f & %.1f \\\\" % (name, n, rank, cond, sd, sv))
    lines += [
        r"\midrule",
        r"six charge states, one line each & 6 & 6 of 7 & 1.0 & \multicolumn{2}{c}{null projection 0} \\",
        r"four lines, three states (earlier set) & 4 & 3 of 7 & --- & \multicolumn{2}{c}{null projection 0.756} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote %s with %d rows" % (OUT, len(rows) + 2))
    for name, n, rank, cond, sd, sv in rows:
        print("  %-22s n=%2d rank=%d k=%6.0f  sigma_D=%5.1f%%  sigma_V=%5.1f%%" % (name, n, rank, cond, sd, sv))


if __name__ == "__main__":
    main()
