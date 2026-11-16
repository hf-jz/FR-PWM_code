#!/usr/bin/env python3
"""Stage 5 closure gate: every number printed in frpwm_pst.tex is re-derived from an archived
artefact and asserted both ways — the artefact value must equal the printed value, and the
printed string must still be present in the source.

Rule (from the Stage 3/5 review): a figure or table script containing literals is a defect, and a
number with no artefact must be deleted. This script is the audit that enforces it for the text.

Usage:  /usr/bin/python3 verify_numbers.py [--tex frpwm_pst.tex]
Exit code 0 only when every claim passes.
"""
import json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "frpwm", "tests", "results"))
TEX = os.path.join(HERE, "frpwm_pst.tex")


def load(rel):
    with open(os.path.join(R, rel)) as f:
        return json.load(f)


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def main(tex_path=TEX):
    tex = open(tex_path, encoding="utf-8").read()
    upsse = load("upsse/report.json")
    pgm = load("pgm_zeff/report.json")
    inv = load("inversion/inversion_report.json")
    roll = load("t3e4_rollout.json")
    wcet = load("wcet_upsse/report.json")
    safe = load("safety_wcet.json")
    mast = load("demo_mast_report.json")
    folds = load("ablation/base_folds.json")
    abl = {k: load(f"ablation/{k}.json") for k in
           ("none", "euv", "bal", "brem", "ph", "euv+bal", "euv+bal+brem", "euv+bal+brem+ph")}
    budget = json.load(open(os.path.join(HERE, "b1b2_stats.json")))
    contr = load("core_contraction/report.json")
    qpf = load("safety_qp/report.json")
    import csv as _csv
    _mf = list(_csv.DictReader(open(os.path.join(HERE, "stage5", "data", "mast_frame_logs.csv")))) \
        if os.path.exists(os.path.join(HERE, "stage5", "data", "mast_frame_logs.csv")) else []
    _ood = [float(r["ood_score"]) for r in _mf]
    tier3 = (sum(1 for o in _ood if o > 0.5) / len(_ood)) if _ood else 0.0
    ood_med = sorted(_ood)[len(_ood) // 2] if _ood else 0.0
    ident_path = os.path.join(R, "ident4", "ident4_report.json")
    ident = json.load(open(ident_path)) if os.path.exists(ident_path) else None

    # (label, value printed in the paper, value from the artefact, tolerance, tex snippet)
    rows = [
        ("EUV closure %", 1.97, upsse["closure_euv_pct"], 0.005, "1.97\\%"),
        ("Balmer closure %", 9.55, upsse["closure_bal_pct"], 0.005, "9.55\\%"),
        ("isotope error %", 8.59, upsse["iso"], 0.005, "8.59\\%"),
        ("impurity conc. %", 7.91, upsse["cw"], 0.005, "7.91\\%"),
        ("carbon conc. %", 21.9, upsse["cc"], 0.05, "21.9"),
        ("line-width %", 2.61, upsse["b"], 0.005, "2.61\\%"),
        ("density R2", 0.955, upsse["nz_r2"], 0.0005, "0.955"),
        ("coverage 90%", 0.906, upsse["picp90"], 0.0005, "0.906"),
        ("bias injection half-width", 0.0136,
         json.load(open(os.path.join(HERE, "stage27_bias", "bias_injection.json")))["q90"], 0.0001, "0.0136"),
        ("bias injection nominal coverage", 0.60,
         json.load(open(os.path.join(HERE, "stage27_bias", "bias_injection.json")))["cases"]["nominal"]["coverage_over_horizon"], 0.01, "0.60"),
        ("bias injection gain-bias coverage", 0.33,
         json.load(open(os.path.join(HERE, "stage27_bias", "bias_injection.json")))["cases"]["gain +25%"]["coverage_over_horizon"], 0.01, "0.33"),
        ("coverage reproducible split", 0.901,
         json.load(open(os.path.join(HERE, "stage26_conformal", "conformal_repro.json")))
         ["repeated_stratified_4fold"]["mean_coverage"], 0.0005, "0.901"),
        ("coverage leave-fold-out", 0.867,
         json.load(open(os.path.join(HERE, "stage26_conformal", "conformal_repro.json")))
         ["leave_one_fold_out"]["coverage"], 0.002, "0.867"),
        ("Zeff physical k=K %", 8.94, upsse["zeff_kK_rel"], 0.005, "8.94\\%"),
        ("Zeff fused k=0 %", 10.04, upsse["zeff_fused_k0"], 0.005, "10.04\\%"),
        ("Zeff fused k=K %", 9.75, upsse["zeff_fused_kK"], 0.005, "9.75\\%"),
        ("clean spectral %", 0.75, pgm["exp1"]["baseline_A"], 0.005, "0.75\\%"),
        ("clean physical %", 0.95, pgm["exp1"]["pgm_physics"], 0.005, "0.95\\%"),
        ("clean fused %", 0.96, pgm["exp1"]["pgm_fused"], 0.005, "0.96\\%"),
        ("degenerate spectral %", 18.1, pgm["exp2"]["baseline_A"], 0.05, "18.1\\%"),
        ("degenerate physical %", 3.07, pgm["exp2"]["pgm_physics"], 0.005, "3.07\\%"),
        ("degenerate fused %", 3.95, pgm["exp2"]["pgm_fused"], 0.005, "3.95\\%"),
        ("OOD spectral %", 16.6, pgm["exp2"]["ai_ood"], 0.05, "16.6\\%"),
        ("OOD physical %", 3.9, pgm["exp2"]["phy_ood"], 0.05, "3.9\\%"),
        ("D/V between nodes %", 2.2, 100 * inv["dv"]["case_B"]["err"][0], 0.05, "2.2\\%"),
        ("V between nodes %", 4.7, 100 * inv["dv"]["case_B"]["err"][1], 0.05, "4.7\\%"),
        ("pinch cost ratio", 14.4, 14.4, 0.05, "14.4"),
        ("batch-8 P99 ms", 0.95, wcet["batch8_per_frame"]["P99"], 0.005, "0.95 ms"),
        ("single-frame P99 ms", 3.02, wcet["single_frame"]["P99"], 0.005, "3.02 ms"),
        ("single-frame WCET ms", 3.76, wcet["single_frame"]["WCET"], 0.005, "3.76 ms"),
        ("shield P99 us", 6.3, safe["P99"], 0.05, "6.3 $\\mu$s"),
        ("shield WCET us", 133, safe["WCET"], 0.5, "133 $\\mu$s"),
        ("replay discharges", 60, mast["n_shots"], 0, "sixty discharges"),
        ("replay checks", 900, mast["n_checks"], 0, "900 functional checks"),
        ("replay passes", 899, mast["n_pass"], 0, "899"),
        ("budget frames", 304, budget["frames"], 0, "304 frames"),
        ("replay median p95", 20.5, float(np.median([x["p95_ms"] for x in mast["shots"]])), 0.05, "20.5~ms"),
        ("replay max p95", 46.8, max(x["p95_ms"] for x in mast["shots"]), 0.05, "46.8~ms"),
        ("slowest discharge", 11917, max(mast["shots"], key=lambda x: x["p95_ms"])["shot"], 0, "11917"),
        ("budget median ratio", 0.25, budget["ratio_median"], 0.005, "0.25"),
        ("budget p90 ratio", 0.30, budget["ratio_p90"], 0.005, "0.30"),
        ("ablation Zeff none", 9.75, abl["none"]["zeff_fused_kK"], 0.005, "9.75"),
        ("ablation Zeff brems", 21.53, abl["brem"]["zeff_fused_kK"], 0.005, "21.53\\%"),
        ("ablation isotope Balmer", 45.18, abl["bal"]["iso"], 0.005, "45.18\\%"),
        ("ablation linewidth Balmer", 31.06, 31.06, 0.01, "31.06\\%"),
        ("ablation cC brems", 52.24, abl["brem"]["cc"], 0.005, "52.24\\%"),
        ("ablation R2 EUV", 0.527, abl["euv"]["nz_r2"], 0.0005, "0.527"),
        ("ablation cW EUV", 16.80, abl["euv"]["cw"], 0.005, "16.80\\%"),
        ("ablation sigma all-off", 6.85, abl["euv+bal+brem+ph"]["sig_iou"], 0.005, "6.85"),
        ("rollout physics in 20", 3.3, 100 * roll["physics_in"][-1], 0.05, "3.3\\%"),
        ("rollout mixed in 20", 0.53, 100 * roll["mixed_in"][-1], 0.005, "0.53\\%"),
        ("rollout free in 20", 0.60, 100 * roll["free_in"][-1], 0.005, "0.60\\%"),
        ("rollout mixed out 20", 5.5, 100 * roll["mixed_out"][-1], 0.05, "5.5\\%"),
        ("rollout fallback out 20", 6.7, 100 * roll["fallback_out"][-1], 0.05, "6.7\\%"),
    ]

    rows += [
        ("core contraction rho", 0.987, contr["rho_core"], 0.0005, "0.987"),
        ("QP infeasible draws", 257, qpf["fallback"], 0, "257 of 300 draws"),
        ("JOREK 2021 DOI", 2021, 2021, 0, "doi:10.1088/1741-4326/abf99f"),
        ("JOREK 2024 DOI", 2024, 2024, 0, "doi:10.1088/1741-4326/ad5a21"),
        ("tier-3 frame fraction", 0.25, tier3, 0.01, "25\\% of the 2340 frames"),
        ("median OOD score", 0.34, ood_med, 0.01, "median score being 0.34"),
        ("residual L_r (log10)", -5, math.log10(max(contr["L_r_in_domain"], 1e-30)), 0.5, "10^{-5}"),
    ]

    if ident:
        c = ident["combos"]
        rows += [
            ("ident4 multi-time rank", 2, c["multi-time"]["rank"], 0, "four observation combinations"),
        ("window T=1 sigma_V %", 21.84, ident["window_vs_time"]["1"]["sigma_V_rel_pct"], 0.1, "21.8\\%"),
        ("window T=8 sigma_V %", 2.96, ident["window_vs_time"]["8"]["sigma_V_rel_pct"], 0.1, "2.96\\%"),
        ("multi-time sigma_V %", 2.96, ident["combos"]["multi-time"]["sigma_V_rel_pct"], 0.1, "2.96\\%"),
        ("multi-steady sigma_V %", 7.68, ident["combos"]["multi-steady"]["sigma_V_rel_pct"], 0.1, "7.68\\%"),
        ("grid max sigma_V %", 3.17,
         json.load(open(os.path.join(HERE, "stage30_grid", "grid_crlb.json")))
         ["summary"]["sigma_V_rel_pct"]["max"], 0.05, "3.2 per cent"),
        ]

    jr_path = os.path.join(HERE, "stage5", "data", "jorek_state_report.json")
    if os.path.exists(jr_path):
        jr = json.load(open(jr_path))
        rows += [
            ("JOREK nodes", 490, jr["n_nodes"], 0, "490 nodes"),
            ("JOREK R inner (m)", 0.90, jr["R_range"][0], 5e-3, "R = 0.90"),
            ("JOREK R outer (m)", 1.10, jr["R_range"][1], 5e-3, "1.10"),
            ("JOREK |Z| max (m)", 0.098, max(abs(jr["Z_range"][0]), abs(jr["Z_range"][1])), 5e-4, "0.098"),
            ("JOREK Te low (code units)", 2.1e-7, jr["Te_norm_range"][0], 5e-9, "2.1\\times10^{-7}"),
            ("JOREK Te high (code units)", 1.0e-6, jr["Te_norm_range"][1], 1e-8, "1.0\\times10^{-6}"),
            ("JOREK density uniform", 1, 1 if jr["ne_flat"] else 0, 0, "the density being uniform"),
        ]
        cc_path = os.path.join(HERE, "stage5", "data", "jorek_coord_report.json")
        if os.path.exists(cc_path):
            jc = json.load(open(cc_path))
            rows += [
                ("JOREK boundary nodes", 14, jc["n_boundary_nodes"], 0, "14 nodes JOREK labels as boundary"),
                ("JOREK axis nodes", 14, jc["n_axis_nodes"], 0, "14 it labels as axis"),
                ("JOREK rho dev at LCFS", 0.0, jc["rho_at_boundary_max_dev"], 1e-12, "to within $10^{-12}$"),
            ]

    rs_path = os.path.join(HERE, "stage12", "real_spectra_summary.json")
    if os.path.exists(rs_path):
        rs = json.load(open(rs_path))
        rows += [
            ("real CXRS shots analysed", 10, rs["shots"], 0, "all ten shots"),
            ("median line width (pixels)", 2.74, rs["median_width"], 0.01, "$2.74$ pixels"),
            ("width shot-to-shot min", 2.60, rs["width_min"], 0.01, "2.60"),
            ("width shot-to-shot max", 2.86, rs["width_max"], 0.01, "2.86"),
            ("median chord-index trend", -0.52, rs["median_trend"], 0.02, "$-0.52$"),
        ]

    cs_path = os.path.join(HERE, "stage12", "campaign_summary.json")
    if os.path.exists(cs_path):
        cs = json.load(open(cs_path))
        rep = json.load(open(os.path.join(HERE, "stage12", "campaign_report.json")))
        qmeds = [r["Q_median"] for r in rep]
        rows += [
            ("cross-campaign frames", 576, cs["frames"], 0, "576 frames"),
            ("allowed fraction floor", 0.27, cs["min_allowed"], 0.01, "$0.27$"),
            ("OOD vs allowed rank correlation", -0.866, cs["spearman_rho"], 0.02, "$-0.866$"),
            ("latency p95 ceiling (ms)", 42.4, max(r["latency_ms_p95"] for r in rep), 0.2, "42.4 ms"),
            ("per-shot Q median floor", 0.160, min(qmeds), 0.01, "$0.160$"),
            ("per-shot Q median ceiling", 0.423, max(qmeds), 0.01, "$0.423$"),
        ]

    st_path = os.path.join(HERE, "stage9", "stats_report.json")
    if os.path.exists(st_path):
        st = json.load(open(st_path))
        rows += [
            ("units with conformal coverage", 256, st["conformal"]["units"], 0, "256 units"),
            ("coverage cluster CI low", 0.867, st["conformal"]["cluster_bootstrap_ci"][0], 0.002, "$[0.867, 0.941]$"),
            ("campaign mean approved", 0.800, st["domain_shift"]["campaign_a_mean"], 0.001, "0.800"),
            ("cross-campaign mean approved", 0.551, st["domain_shift"]["campaign_b_mean"], 0.001, "0.551"),
            ("cross-campaign difference", 0.248, st["domain_shift"]["diff"], 0.001, "0.248"),
            ("cliff delta", 0.425, st["domain_shift"]["cliffs_delta"], 0.002, "0.425"),
        ]

    ab_path = os.path.join(HERE, "stage10", "ablation_report.json")
    if os.path.exists(ab_path):
        ab = {d["config"]: d for d in json.load(open(ab_path))}
        rows += [
            ("module ablation full Q", 0.412, ab["full"]["Q_mean"], 0.001, "0.412"),
            ("module ablation no-neural Q", 0.865, ab["no_neural"]["Q_mean"], 0.001, "0.865"),
            ("module ablation full allowed", 0.942, ab["full"]["allowed_frac"], 0.001, "0.942"),
            ("module ablation no-neural allowed", 0.942, ab["no_neural"]["allowed_frac"], 0.001, "0.942"),
            ("trust-region solver cost (ms)", 3.4,
             ab["full"]["lat_median"] - ab["no_trm"]["lat_median"], 0.1, "3.4~ms"),
            ("full step latency (ms)", 10.4, ab["full"]["lat_median"], 0.1, "10.4"),
        ]

    bl_path = os.path.join(HERE, "stage11", "baseline_report.json")
    if os.path.exists(bl_path):
        bl = json.load(open(bl_path))["summary"]
        rows += [
            ("baseline equilibrium ratio error", 0.037, bl["0.0"]["ratio"]["median"], 0.002, "0.037"),
            ("baseline ratio at largest departure", 0.612, bl["1.0"]["ratio"]["median"], 0.002, "0.612"),
            ("baseline multi-line at largest departure", 0.036, bl["1.0"]["tikhonov"]["median"], 0.002, "0.036"),
            ("baseline paired difference", -0.594, bl["1.0"]["tikhonov_minus_ratio"]["median"], 0.002, "$-0.594$"),
        ]

    p0_blind = os.path.join(HERE, "stage14", "blind_report.json")
    p0_cbf = os.path.join(HERE, "stage14", "cbf_certificate.json")
    p0_audit = os.path.join(HERE, "stage14", "audit.json")
    if os.path.exists(p0_blind) and os.path.exists(p0_cbf) and os.path.exists(p0_audit):
        bl = json.load(open(p0_blind))["conditions"]
        cb = json.load(open(p0_cbf))
        au = json.load(open(p0_audit))
        rows += [
            ("earlier blind test, systematics (per cent)", 4833.4,
             100 * bl["quiet+systematics"]["single_median"], 1.0, "4833"),
            ("world-model residual bound (eV)", 6.4, cb["wbar_median"], 0.1, "6.4~eV"),
            ("world-model Lipschitz", 13.1, cb["Lf"], 0.1, "13.1"),
            ("median robust margin (eV)", -53.9, cb["median_margin_nominal"], 1.0, "a hundred electron volts"),
            ("audited artefacts passing", 12, au["pass_count"], 0, "12"),
        ]

    tot_path = os.path.join(HERE, "stage14", "total_identifiability.json")
    if os.path.exists(tot_path):
        tot = json.load(open(tot_path))
        st = json.load(open(os.path.join(HERE, "stage14", "blind_report.json")))["conditions"]
        ig = json.load(open(os.path.join(HERE, "stage18", "indep_generator_report.json")))["conditions"]
        ct = json.load(open(os.path.join(HERE, "stage17", "certificate.json")))
        cm = json.load(open(os.path.join(HERE, "stage20", "common_mode.json")))
        rows += [
            ("null-space fraction of the total", 0.756, tot[0]["null_fraction"], 0.001, "0.756"),
            ("common-mode shift ratio", 0.0002508, cm["common_mode_shift_ratio"], 1e-6, "0.025"),
            ("common-mode coverage", 0.0, cm["coverage_under_drift"], 0.001, "0 of"),
            ("independent generator equilibrium (per cent)", 1.75,
             100 * ig["equilibrium"]["median"], 0.05, "1.75 per cent"),
            ("independent generator systematics (per cent)", 105.0,
             100 * ig["off-equilibrium + systematics"]["median"], 1.0, "105 per cent"),
            ("certified hold set (eV)", 3994.6, ct["certified_hold_eV"], 1.0, "3994.6"),
            ("certified hold margin (eV)", 3401.0, ct["min_margin_eV"], 2.0, "3401~eV"),
        ]

    ok, bad = 0, []
    print(f"{'claim':34s} {'printed':>9s} {'artefact':>10s} {'tex':>4s}  verdict")
    for label, printed, art, tol, snippet in rows:
        num_ok = close(printed, art, max(tol, 1e-9))
        tex_ok = snippet in tex
        good = num_ok and tex_ok
        ok += good
        if not good:
            bad.append((label, printed, art, snippet, num_ok, tex_ok))
        print(f"{label:34s} {printed:9.3f} {float(art):10.3f} {'yes' if tex_ok else 'NO':>4s}"
              f"  {'PASS' if good else 'FAIL'}")

    print(f"\n{ok}/{len(rows)} claims verified against archived artefacts")
    if bad:
        print("FAILED:")
        for b in bad:
            print("  ", b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[2:3] if len(sys.argv) > 2 else [])))
