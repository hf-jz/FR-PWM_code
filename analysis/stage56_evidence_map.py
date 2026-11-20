"""Every conclusion in the paper, with the file it rests on and the command that regenerates that file.

The number gates of the submission check that a printed number equals an archived number. That is necessary and not
sufficient: a sentence can be numerically accurate and still rest on nothing that exists. This map is the other half.
Each conclusion is listed with the artefact it rests on, the script that produces the artefact, and the command that
reproduces it, and the check here fails if any of the three is missing. The map is written to submission/EVIDENCE.md
and to a JSON file that ships with the code repository, whose rerun tool reads it.

Writes submission/EVIDENCE.md and submission/evidence_map.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (id, conclusion as it appears in the paper, artefact, key inside the artefact, producing script, extra deps)
CLAIMS = [
    ("E1", "The six-line observation set leaves a one-dimensional null space and the total concentration is orthogonal to it.",
     "stage16/sixline_report.json", "rank", "stage16_sixline.py", ""),
    ("E2", "Four lines covering three charge states give rank 3 of 7 and a null-space projection of 0.756 on the total.",
     "stage14/total_identifiability.json", "null_projection", "stage14_total_ident.py", ""),
    ("E3", "Closure error of the four line-based targets is 1.97 to 9.55 per cent against the frozen band.",
     "stage5/coverage.json", "figure3_closure.csv.rows", "stage5/build_tables.py", ""),
    ("E4", "Eight of nine transport-code comparisons fall inside the predefined band.",
     "stage5/strahl_rows_summary.json", "c1 基线.median_ratio", "stage5/strahl_rows.py", "STRAHL_CPP"),
    ("E5", "Repeated stratified four-fold coverage is 0.901 with a standard deviation of 0.006.",
     "stage26_conformal/conformal_repro.json", "mean_coverage", "stage26_conformal_repro.py", ""),
    ("E6", "Leave-one-fold-out coverage is 0.867, which is the honest number for a single split.",
     "stage26_conformal/conformal_repro.json", "leave_one_fold_out", "stage26_conformal_repro.py", ""),
    ("E7", "The independently written generator reproduces equilibrium cases to 1.75 per cent and degrades to 40.12 and 105.34 per cent.",
     "stage18/indep_generator_report.json", "off_equilibrium_pct", "stage18_indep_gen.py", ""),
    ("E8", "The protection action keeps the certified inner set invariant, with the residual bounded over the box by interval propagation.",
     "stage53/certified_bound.json", "boxes.physical band 150-4000 eV.abs_bound_eV", "stage53_certified_bound.py", ""),
    ("E9", "No tracking action certifies on any of the 120 frames, the median nominal margin being -53.9 eV.",
     "stage14/cbf_certificate.json", "nominal_median_margin_eV", "stage14_p0_cbf.py", ""),
    ("E10", "The QP is infeasible in 257 of 300 draws and every one returns the lookup command.",
     "stage14/audit.json", "pass_count", "stage14_p0_audit.py", ""),
    ("E11", "Replay releases 80 per cent of 2400 frames, and the post-lock check passes 142 of 150 on the ten discharges.",
     "stage40/postfreeze_checks.json", "passed", "stage40_postlock_checks.py", "FAIR-MAST cache"),
    ("E12", "Bias injection leaves the interval half-width at 0.0136 while recursive coverage falls from 0.60 to 0.33.",
     "stage27_bias/bias_injection.json", "nominal_coverage", "stage27_bias_injection.py", ""),
    ("E13", "The nine contract scenarios return supported, withheld, degraded and invalid as specified, and the ignored bias is wrong by 12.45 per cent.",
     "stage44/target_contract.json", "ignored_bias_error_pct", "stage44_target_contract.py", ""),
    ("E14", "Fitted line width tracks the analysed ion temperature with a Spearman coefficient of 0.79 against a permuted 0.11.",
     "stage48/width_vs_ti.json", "spearman", "stage48b_width_vs_ti.py", "FAIR-MAST spectra"),
    ("E15", "With real diagnostic input the chain takes 8.7 ms at the median and 77.6 ms at the 95th percentile per frame.",
     "stage48/real_crosscheck.json", "timing_real_input.p95", "stage48_real_crosscheck.py", "FAIR-MAST cache"),
    ("E16", "Over loopback the shield and interface take 0.45 ms at the median, and the pacing layer misses 95 per cent of ticks.",
     "stage50/loopback_hil.json", "deadline_miss_rate", "stage50_loopback_hil.py", ""),
    ("E17", "The same bundle replays the same frames to the same digits in separate processes on this stack.",
     "stage55/determinism.json", "runs.28315.identical", "stage55_determinism.py", "FAIR-MAST cache"),
    ("E18", "The instrument width is 2.74 px at the median, and a 1.69 fold width error changes the ratio by 41 and 84 per cent.",
     "stage31/instrument_width.json", "median_px", "stage31_instrument_width.py", "FAIR-MAST spectra"),
    ("E19", "The replay plant accepts a screened command and the recorded profile does not move, so the plant cannot be actuated.",
     "stage57/closed_loop.json", "by_band.frozen 150-8000 eV.28315.released", "stage57_closed_loop.py", "FAIR-MAST cache"),
    ("E20", "Against the world model as plant, no command passes at five per cent uncertainty and commands begin to pass at 0.5 per cent.",
     "stage59/model_loop.json", "eps_sweep.3.eps_rel", "stage59_model_loop.py", ""),
]


def dig(obj, path):
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list):
            obj = obj[int(part)]
        else:
            return None
    return obj


rows, missing = [], []
for cid, claim, art, key, script, deps in CLAIMS:
    ap = os.path.join(HERE, art)
    sp = os.path.join(HERE, script)
    val, ok_art = None, os.path.exists(ap)
    if ok_art and art.endswith(".json"):
        try:
            val = dig(json.load(open(ap)), key)
        except Exception:  # noqa: BLE001
            val = None
    ok_script = os.path.exists(sp)
    rows.append(dict(id=cid, claim=claim, artefact=art, key=key, value=val, script=script,
                     command="python3 %s" % script, deps=deps,
                     artefact_present=ok_art, script_present=ok_script))
    if not (ok_art and ok_script):
        missing.append((cid, art, ok_art, script, ok_script))

map_path = os.path.join(HERE, "submission", "evidence_map.json")
json.dump(dict(claims=rows, all_artefacts_present=not missing), open(map_path, "w"), indent=1)

md = ["# Evidence map", "",
      "Each conclusion the paper draws, the file it rests on, the script that makes that file, and the command that",
      "reproduces it. A conclusion whose artefact or script is absent is a conclusion the paper may not draw.", "",
      "| id | conclusion | artefact | value | command |", "|---|---|---|---|---|"]
for r in rows:
    v = "—" if r["value"] is None else ("%g" % r["value"] if isinstance(r["value"], (int, float)) else str(r["value"]))
    md.append("| %s | %s | `%s` (%s) | %s | `%s` |" % (r["id"], r["claim"], r["artefact"], r["key"], v, r["command"]))
md += ["", "External data needed for a rerun is named in the last column of the JSON map; scripts that need none run on a "
      "bare interpreter with numpy.", ""]
open(os.path.join(HERE, "submission", "EVIDENCE.md"), "w").write("\n".join(md))

print("%d conclusions mapped, %d with artefacts and scripts present" % (len(rows), len(rows) - len(missing)))
for m in missing:
    print("MISSING", m)
assert not missing, "a conclusion has no artefact or no script"
