#!/usr/bin/env python3
"""Tables 6 and 7 of the manuscript, from artefact A22.

table_sdtif.tex        analytic benchmarks and the applied grades
table_carbon_nulls.tex per-node evidence: unconditional and conditional carbon inversion
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT  # noqa: E402

A22 = os.path.join(ARTEFACTS, "A22_SD_TIF")


def fmt(x, spec="%.4f", dash="---"):
    return dash if x is None else spec % x


def main():
    b = json.load(open(os.path.join(A22, "benchmarks.json")))
    c = json.load(open(os.path.join(A22, "cases.json")))
    os.makedirs(OUT, exist_ok=True)
    a, t, g, e = c["case_A_unconditional"], c["case_D_transport"], c["case_C_gain"], c["case_E_map"]
    f = c["case_F_realdata"]
    fb = f["calibration_box"]["targets"]
    fs = f["noise_sweep"]
    ms = f["map_summary"]
    row0 = a[0]
    e1, e2 = b["E1 linear, two null directions"], b["E2 nonlinear, local rank change"]
    e3, e4 = b["E3 nuisance calibration parameter"], b["E4 state sweep, rank transition"]
    e6 = b["E6 Weyl / Davis-Kahan robustness"]
    e7 = b["E7 analytic bound, independent projector"]
    e9 = b["E9 edge cases"]
    c2 = c["case_D_sources"]
    gl = g["singular_values_lines_only"]
    gw = g["singular_values_with_interferometer"]
    sw = g["relative_sigma_with_interferometer"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{The calculus against closed-form answers and applied, from artefact A22. "
        r"$\Gamma$ is the fraction of the target gradient in the unobservable subspace, so "
        r"$\Gamma=0$ means the observations fix the target to first order and $\Gamma>0$ means "
        r"they cannot at any signal-to-noise.}",
        r"\label{tab:sdtif}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        r"case & analytic & computed & verdict \\",
        r"\midrule",
        r"linear, $\Gamma(q_{13})$, rank 2 & %.4f & %.4f & $2^{-1/2}$ \\" % (2 ** -0.5, e1["gamma"]["q13"]),
        r"nonlinear, singular point, rank & 1 & %d & lost on $\theta_2=2\theta_1^2$ \\" % e2["degenerate_rank"],
        r"nonlinear, $\Gamma(\theta_1)$ there & %.4f & %.4f & $1/\sqrt5$ \\" % (5 ** -0.5, e2["gamma_degenerate"]["th1"]),
        r"nuisance gain, $\Gamma(n_z)$ gain known & 0 & %.1e & identifiable \\" % e3["gamma_fixed"],
        r"nuisance gain, $\Gamma(n_z)$ gain free & %.4f & %.4f & profiled information 0 \\" % (2 ** -0.5, e3["gamma_free"]),
        r"rank transition, certified away & --- & %s & a 2\,\%% box is covered \\"
        % ("yes" if e4["certified_away_from_crossing"] else "no"),
        r"rank transition, certified across & --- & %s & refused, as it must be \\"
        % ("yes" if e4["certified_across_crossing"] else "no"),
        r"$\|P_0'-P_0\|$, $\delta_J=2.0$, gap 95 & $\leq%.4f$ & %.4f & holds across the sweep \\"
        % (e6["P_bound"], e6["unobservable_subspace_drift"]),
        r"pseudo-inverse on a pure-null target & --- & %.1f & not a bound (E9) \\"
        % e9["target_in_null"]["pseudo_pure"],
        r"bound on a sum of two, $c^{\mathsf T}\mathcal F^{+}c$ & %.1e & %.1e & closed form, $2\sigma^2$ \\"
        % (e7["bound_analytic"], e7["bound_computed"]),
        r"null projector, second code path & 0 & %.1e & eigendecomposition vs SVD \\"
        % e7["projector_independent_path"],
        r"\midrule",
        r"transport, $\Gamma(D)$, source amplitude free & %.4f & %.4f & $1/\sqrt3$ \\"
        % (3 ** -0.5, t["targets"]["D0"]["gamma"]),
        r"transport, $\Gamma(D/V)$ & 0 & %.1e & the ratio survives the symmetry \\"
        % t["targets"]["D_over_V"]["gamma"],
        r"three lines, unknown gain, $\sigma_2/\sigma_1$ & --- & %.1e & density--gain separation \\"
        % (gl[1] / gl[0]),
        r"idem, $\sigma(n_e)$ (\%% with interferometer) & --- & %.2f & from %.1e to %.1f \\"
        % (100 * sw["n_e"], gl[1], gw[1]),
        r"real MAST 28315, $\Gamma(T_i)$ gain free & %.4f & %.4f & amplitude tied to calibration \\"
        % (2 ** -0.5, f["targets"]["T_absolute"]["gamma"]),
        r"idem, $\Gamma(\ln L_T)$ gain free & 0 & %.1e & $\sigma=%.2f$\%% at %.1f\%% noise \\"
        % (f["targets"]["ln_L_T"]["gamma"], 100 * f["targets"]["ln_L_T"]["sigma"], f["sigma_pct"]),
        r"idem, 15\%% calibration error on $\ln L_T$ & 0 & %.1e & worst corner \\"
        % fb["ln_L_T"]["gamma_max"],
        r"idem, noise multiplier certified & --- & %.2f & refused above it \\"
        % max(b2 for _, b2 in fs["certified"]) if fs["certified"] else
        (r"idem, noise multiplier certified & --- & none & refused at every level \\"),
        r"continuous certificate, $n_e$ axis & $\delta_J<g/2$ & %.1f$<$%.1f & empirical level \\"
        % (e["certificate_ne"]["delta_max"], e["certificate_ne"]["g_min"]),
        r"transport, measured source, $\Gamma(D)$ & 0 & %.1f & the data break the symmetry \\"
        % c2["measured_source"]["targets"]["D"]["gamma"],
        r"campaign 28315--28324, $\Gamma(T_i)$ / $\Gamma(\ln L_T)$ & $2^{-1/2}$ / 0 & "
        r"%.0f / %.0f of %.0f & certified in %.0f \\"
        % (ms["amplitude_degenerate"], ms["shape_identifiable"], ms["n_shots"],
           ms["shape_certified_at_nominal"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    open(os.path.join(OUT, "table_sdtif.tex"), "w").write("\n".join(lines) + "\n")

    rows = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{The same audit on each shot of the campaign, from artefact A22. "
            r"$\Gamma(T_i)=2^{-1/2}$ in every shot because the amplitude is degenerate with the "
            r"channel calibration; $\Gamma(\ln L_T)=0$ in every shot; $\mu$ is the noise multiplier "
            r"at which the gradient-length bound would leave the required accuracy, so a blank entry "
            r"means the certificate is refused at the measured noise.}",
            r"\label{tab:mastmap}", r"\small",
            r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"shot & $t$ (s) & radii & $\sigma/\sigma_0$ (\%) & $\Gamma(T_i)$ & "
            r"$\sigma(\ln L_T)$ (\%) & $\mu$ \\", r"\midrule"]
    for m in f["map"]:
        rows.append(r"%d & %.3f & %d & %.1f & %.4f & %.3f & %s \\"
                    % (m["shot"], m["time"], m["n_radii"], m["sigma_pct"], m["gamma_T"],
                       100 * m["sigma_L"] if m["sigma_L"] else 0.0,
                       ("%.1f" % m["certified_mu"]) if m["certified_mu"] >= 1.0 else "---"))
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "table_mast_map.tex"), "w").write("\n".join(rows) + "\n")

    # ------------------------------------------- equilibrium-weighted operator, per node
    w = c["case_A_weighted"]
    rows = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{The equilibrium-weighted six-by-seven operator on the same grid, from artefact A22. "
            r"$T_e$ in eV, $n_e$ in m$^{-3}$, occupied the number of stages carrying population. The null "
            r"space has more than one dimension wherever the weighted columns are rank deficient, so no "
            r"single vector is printed: the full basis and the projector of every node are in the record.}",
            r"\label{tab:nullspace}", r"\small",
            r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
            r"$T_e$ & $n_e$ & occupied & rank & $\dim\,\mathrm{Null}$ & $\Gamma(N_{\mathrm{tot}})$ & "
            r"$\Gamma(N_{\mathrm{obs}})$ \\", r"\midrule"]
    for row in w:
        rows.append(r"%.0f & %.0e & %d & %d & %d & %.3f & %.3f \\"
                    % (row["te"], row["ne"], row["occupied"], row["rank_num"],
                       len(row["null_basis"]), row["targets"]["N_tot"]["gamma"],
                       row["targets"]["N_obs"]["gamma"]))
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "table_weighted_nulls.tex"), "w").write("\n".join(rows) + "\n")

    # --------------------------------------------------------------- carbon table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Carbon, per node, from artefact A22. (a) Unconditional inversion: seven free "
        r"charge-state densities from six lines, column $k$ the emissivity of the line of stage $k$. "
        r"The operator is diagonal, so with three per cent relative noise on each line every observed "
        r"stage is equally conditioned and the seventh singular value is an exact zero at every node "
        r"because no line of the fully stripped stage exists; the structural rank is six and the "
        r"structural null space one-dimensional everywhere. (b) Conditional inversion: total "
        r"concentration and temperature from the same lines, populations following the ionisation "
        r"balance. $\Gamma=0$ at every node, so the concentration is determined wherever the data "
        r"resolve it; a dagger marks a bound computed over a subspace that does not span the whole "
        r"target gradient, hence a lower bound rather than the target's uncertainty. (c) The same "
        r"lines with the equilibrium fractions folded into the columns, as in the earlier draft: the "
        r"unknown is a scale per stage, a stage with no population leaves a zero column, and the rank "
        r"is the count of populated and observed stages.}",
        r"\label{tab:carbon}",
        r"\small",
        r"\begin{tabular}{@{}rrrrrrl@{}}",
        r"\toprule",
        r"$T_e$ (eV) & rank$_{\mathrm{str}}$ & rank$_{\mathrm{prac}}$ & $\sigma_7/\sigma_1$ & "
        r"$\Gamma(N_{\mathrm{tot}})$ & $\Gamma(N_{\mathrm{obs}})$ & class of $N_{\mathrm{obs}}$ \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{(a) seven free stage densities, $n_e=10^{14}$~cm$^{-3}$}}\\",
        r"\addlinespace[2pt]",
    ]
    for r in [x for x in a if x["ne"] == 1e14]:
        lines.append(r"%.0f & %d & %d & %.1e & %.4f & %.1e & %s \\" % (
            r["te"], r["rank_num"], r["rank_prac"], r["singular_values"][6] / r["singular_values"][0],
            r["targets"]["N_tot"]["gamma"], r["targets"]["N_obs"]["gamma"],
            r["targets"]["N_obs"]["cls"].replace("_", "-")))
    lines += [
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{(b) conditional, $\theta=(\ln c,\ln T_e)$, target $c$}}\\",
        r"\addlinespace[2pt]",
        r"$T_e$ (eV) & rank$_{\mathrm{str}}$ & rank$_{\mathrm{prac}}$ & $\sigma_2/\sigma_1$ & "
        r"$\Gamma(c)$ & $\sigma(c)$\,\% & class of $c$ \\",
        r"\midrule",
    ]
    for r in [x for x in c["case_B_conditional"] if x["ne"] == 1e14]:
        sg = r["rel_sigma_te_known"]
        lower = r["rank_prac"] < r["rank_num"]          # bound over a partial subspace
        lines.append(r"%.0f & %d & %d & %.1e & %.1e & %s & %s%s \\" % (
            r["te"], r["rank_num"], r["rank_prac"],
            r["singular_values"][1] / r["singular_values"][0],
            r["gamma_c"],
            "---" if sg is None else "%.2f" % (100 * sg),
            r["cls"].replace("_", "-"), r"$^{\dagger}$" if lower else ""))
    lines += [
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{(c) the same lines with the ionisation balance folded into the "
        r"columns, unknown a scale per stage}}\\",
        r"\addlinespace[2pt]",
        r"$T_e$ (eV) & rank$_{\mathrm{str}}$ & rank$_{\mathrm{prac}}$ & occupied & "
        r"$\Gamma(N_{\mathrm{tot}})$ & $\Gamma(N_{\mathrm{obs}})$ & --- \\",
        r"\midrule",
    ]
    for r in [x for x in c["case_A_weighted"] if x["ne"] == 1e14]:
        lines.append(r"%.0f & %d & %d & %d & %.4f & %.4f & --- \\" % (
            r["te"], r["rank_num"], r["rank_prac"], r["occupied"],
            r["targets"]["N_tot"]["gamma"], r["targets"]["N_obs"]["gamma"]))
    lines += [
        r"real MAST 28315, $\Gamma(T_i)$ gain free & %.4f & %.4f & amplitude tied to calibration \\"
        % (2 ** -0.5, f["targets"]["T_absolute"]["gamma"]),
        r"idem, $\Gamma(\ln L_T)$ gain free & 0 & %.1e & $\sigma=%.2f$\%% at %.1f\%% noise \\"
        % (f["targets"]["ln_L_T"]["gamma"], 100 * f["targets"]["ln_L_T"]["sigma"], f["sigma_pct"]),
        r"idem, 15\%% calibration error on $\ln L_T$ & 0 & %.1e & worst corner \\"
        % fb["ln_L_T"]["gamma_max"],
        r"idem, noise multiplier certified & --- & %.2f & refused above it \\"
        % max(b2 for _, b2 in fs["certified"]) if fs["certified"] else
        (r"idem, noise multiplier certified & --- & none & refused at every level \\"),
        r"continuous certificate, $n_e$ axis & $\delta_J<g/2$ & %.1f$<$%.1f & empirical level \\"
        % (e["certificate_ne"]["delta_max"], e["certificate_ne"]["g_min"]),
        r"transport, measured source, $\Gamma(D)$ & 0 & %.1f & the data break the symmetry \\"
        % c2["measured_source"]["targets"]["D"]["gamma"],
        r"campaign 28315--28324, $\Gamma(T_i)$ / $\Gamma(\ln L_T)$ & $2^{-1/2}$ / 0 & "
        r"%.0f / %.0f of %.0f & certified in %.0f \\"
        % (ms["amplitude_degenerate"], ms["shape_identifiable"], ms["n_shots"],
           ms["shape_certified_at_nominal"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    open(os.path.join(OUT, "table_carbon_nulls.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("state axis certified:", e["n_certified_coarse"], "/", len(e["coarse"]),
          "| density axis:", e["n_certified_density_coarse"], "/", len(e["density_coarse"]),
          "->", e["n_certified_density_fine"], "/", len(e["density_fine"]))


if __name__ == "__main__":
    main()
