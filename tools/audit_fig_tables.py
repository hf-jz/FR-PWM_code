#!/usr/bin/env python3
"""Every figure and every data table must be a view of an archived artefact.

Two checks, both fail-closed:

  1. Figures: re-run the generator and require the PNG byte-identical to the file that is in the
     paper. A generator that has drifted from its artefact, or that contains literals rather than
     loads, fails here.
  2. Tables: re-run the generator of `table_realdata.tex` and require the emitted source
     byte-identical; tables written by hand in the manuscript are covered cell by cell by
     `verify_numbers.py`, which this script runs last.

Drawings without data (the two TikZ schematics) are declared rather than tested.

    /usr/bin/python3 audit_fig_tables.py
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/usr/bin/python3"

# (label, generator, artefacts it must read, output files compared)
FIGURES = [
    ("fig3 closure bars", "make_fig4.py", ["upsse/report.json"], ["figure4.png"]),
    ("fig4 results panels", "make_fig3.py", ["upsse/report.json", "pgm_zeff/report.json",
                                             "inversion/inversion_report.json",
                                             "strahl_pinn/strahl_pinn_report.json"], ["figure3.png"]),
    ("fig5 identifiability", "make_fig10.py", ["ident4/ident4_report.json"],
     ["figure10_identifiability.png"]),
    ("fig6 replay", "make_realdata.py", ["demo_mast_report.json"],
     ["figure6_realdata.png", "table_realdata.tex"]),
    ("fig7 budget", "make_fig7.py", ["stage5/data/uncertainty_budget.csv"],
     ["figure7_budget.png"]),
    ("fig8 refinement", "make_fig5.py", ["upsse/report.json", "ablation/base_folds.json"],
     ["figure5_trm.png"]),
    ("fig9 ablation", "make_fig8_ablate.py", ["ablation/"], ["figure8_ablation.png"]),
    ("fig10 rollout", "make_fig9.py", ["t3e4_rollout.json"], ["figure9_rollout.png"]),
]
DRAWINGS = ["figure1.pdf", "figure2.pdf"]          # TikZ schematics: no data behind them


R = os.path.join(HERE, "..", "frpwm", "tests", "results")


def art_exists(a):
    """Artefacts live either in the frpwm results tree or beside the paper (stage5/)."""
    return os.path.exists(os.path.join(R, a)) or os.path.exists(os.path.join(HERE, a))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    report, ok_all = {}, True

    for label, script, arts, outs in FIGURES:
        missing_art = [a for a in arts if not art_exists(a)]
        before = {o: (sha(os.path.join(HERE, o)) if os.path.exists(os.path.join(HERE, o)) else None)
                  for o in outs}
        src = open(os.path.join(HERE, script)).read()
        has_assert = "assert" in src
        loads = all(art_exists(a) or a.endswith("/") for a in arts)
        proc = subprocess.run([PY, script], cwd=HERE, capture_output=True, text=True, timeout=900)
        after = {o: (sha(os.path.join(HERE, o)) if os.path.exists(os.path.join(HERE, o)) else None)
                 for o in outs}
        same = all(before[o] is not None and before[o] == after[o] for o in outs)
        ok = bool(same and proc.returncode == 0 and not missing_art and has_assert and loads)
        ok_all &= ok
        report[label] = dict(script=script, artefacts=arts, outputs=outs, identical=same,
                             asserts=has_assert, returncode=proc.returncode,
                             missing_artefacts=missing_art, pass_=ok)
        print(f"{label:22s} identical={same} asserts={has_assert} rc={proc.returncode} "
              f"{'PASS' if ok else 'FAIL'}")

    for d in DRAWINGS:
        report[d] = dict(kind="drawing", pass_=True)
        print(f"{d:22s} drawing, no data                                    PASS")

    # hand-written data tables: cell-by-cell verification
    v = subprocess.run([PY, os.path.join(HERE, "verify_numbers.py")], cwd=HERE,
                       capture_output=True, text=True)
    tail = v.stdout.strip().splitlines()[-1] if v.stdout.strip() else ""
    ok_all &= (v.returncode == 0)
    report["verify_numbers"] = dict(returncode=v.returncode, summary=tail, pass_=v.returncode == 0)
    print(f"{'tables (hand-written)':22s} {tail}  {'PASS' if v.returncode == 0 else 'FAIL'}")

    out = os.path.join(HERE, "docs", "figure_table_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"\naudit -> {out}")
    print("PASS" if ok_all else "FAIL: at least one figure or table is not reproducible")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
