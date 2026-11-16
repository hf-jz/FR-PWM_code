"""Line-by-line audit of the 2026-11-11 review against the current source.

Each row is one review item, the check that decides it, and where it stands. The point is that the verdict is
computed, not recalled: a grep that finds nothing is a missing item even if the memory of fixing it is vivid.
"""
import io
import re
import sys

TEX = "frpwm_pst.tex"
s = io.open(TEX, encoding="utf-8").read()
abstract = s[s.index("\\noindent\\textbf{Abstract}"):s.index("\\noindent\\textbf{Keywords:}")]
awords = len(abstract.split()) - 2          # the header itself is not prose
# prose is hard-wrapped, so every substring test runs on whitespace-collapsed text
flat = " ".join(s.split())

ROWS = [
    ("P0-1", "chain described as six-line", "per charge state / per ionisation stage",
     (flat.count("one transition per charge state") + flat.count("one transition per ionisation stage")) >= 2),
    ("P0-1", "no stale 'not enabled' claim", "'It is not enabled'", "It is not enabled" not in flat),
    ("P0-1", "no stale 'uses one line' claim", "'uses one line and an equilibrium reference'",
     "uses one line and an equilibrium reference" not in flat),
    ("P0-1", "line labels match ADAS stages", "C I 834.4 present / C II 834.4 absent",
     "C~I 834.4" in flat and "C~II 834.4" not in flat),
    ("P0-1", "rank claim stated with numbers", "rank six, condition number 1.0", "rank six, condition number 1.0" in flat),
    ("P0-2", "three protocols kept apart", "gate-rejection paragraph", "rejection rate of 1.000" in flat),
    ("P0-2", "relative-error denominator defined", "near-zero denominator", "near-zero denominator" in flat),
    ("P0-2", "independent generator separated", "'separate implementation'", "separate implementation" in flat),
    ("P0-3", "0/120 tracking certification reported", "0 per cent + -53.9 eV",
     "0 per cent" in flat and "-53.9" in flat),
    ("P0-3", "protection action proved separately", "'forward invariant'", "forward invariant" in flat),
    ("P0-3", "two safety sets distinguished", "'different sets and are not'", "different sets and are not" in flat),
    ("P0-3", "no 'every command passes'", "phrase absent", "every command passes" not in flat),
    ("P0-4", "criterion is 10 ms", "<10 ms row", "<10$ ms" in s or "<10 ms" in flat),
    ("P0-4", "latency classes named", "batch / kernel / orchestration",
     "batches of eight" in flat and "safety kernel" in s.lower() and "orchestration" in flat),
    ("P0-4", "133 carries the microsecond unit", "133 us present, 133 s absent",
     "133 $\\mu$s" in flat and "133 s" not in flat),
    ("P0-5", "abstract under 300 words", "%d words" % awords, awords < 300),
    ("P0-6", "data availability names delivery", "'supplementary data files'", "supplementary data files" in flat),
    ("P0-6", "no request-only wording", "'reasonable request' absent", "reasonable request" not in flat),
    ("3.1", "conformal guarantee narrowed", "'does not inherit that guarantee'", "does not inherit that guarantee" in flat),
    ("3.2", "fusion called heuristic", "'variance-weighted heuristic fusion'", "variance-weighted heuristic fusion" in flat),
    ("3.2", "no optimality claim", "'minimum-variance fusion' absent", "minimum-variance fusion" not in flat),
    ("3.2", "zero-error caveat kept", "'error against that frame has standard deviation exactly zero'",
     "standard deviation exactly zero" in flat),
    ("3.3", "ablation reads as non-independent", "'cannot serve as an independent check'",
     "cannot serve as an independent check" in flat),
    ("3.3", "refinement does not claim accuracy", "'costs accuracy rather than buying it'",
     "costs accuracy rather than buying it" in flat),
    ("3.4", "MAST scope kept narrow", "pixel-space + synthetic channels",
     "pixel-space" in flat and "synthetic spectroscopic channels" in flat),
    ("3.5", "ADAS cites the manual", "Summers2004", "\\cite{Summers2004}" in flat),
    ("3.5", "JOREK keeps its own citation", "JOREK cite present", "\\cite{Hoelzl2021,Hoelzl2024}" in flat),
    ("3.5", "two Hans disambiguated", "Han Min and Han Mingkun", "Han Min and Han Mingkun" in flat),
    ("3.6", "JOREK check marked non-contributing", "'contributes no result to this paper'",
     "contributes no result to this paper" in flat),
    ("fmt", "title claims no closed-loop control", "'safety-filtered command selection'",
     "safety-filtered command selection" in flat),
    ("fmt", "conclusion lists failures", "'Partly met' and 'Not met'",
     "Partly met" in flat and "Not met" in flat),
]

# ---- the 2026-11-11 14:02 review
ROWS += [
    ("time", "prereg threshold named", "artefact A14, criterion C2", "criterion C2" in flat),
    ("time", "design target separated", "'one tenth' is a target",
     "ne tenth of the control period is a design target" in flat),
    ("3.1", "Tier 2 reports measured coverage", "'attained an empirical coverage of 0.901'",
     "attained an empirical coverage of 0.901" in flat),
    ("3.1", "no 'marginal guarantee delivers'", "phrase absent",
     "marginal guarantee delivers" not in flat),
    ("P0-3", "admissibility vs certification", "'That is admissibility, and only admissibility'",
     "That is admissibility, and only admissibility" in flat),
    ("P0-3", "200/200 not extended to zero-input", "'including the zero-input and hold-last entries'",
     "including the zero-input and hold-last entries" in flat),
    ("audit", "12 of 12 unqualified", "'no un-audited remainder'", "no un-audited remainder" in flat),
    ("4", "blind limits in discussion", "'Identifiability is a property of the observation set'",
     "Identifiability is a property of the observation set" in flat),
    ("4", "blind limits in conclusion", "'Identifiability is not accuracy'",
     "Identifiability is not accuracy" in flat),
]

# ---- the 2026-11-12 review
ROWS += [
    ("time", "no compiled-scheduler prediction", "phrase absent", "compiled scheduler" not in flat),
    ("time", "SIL versus replay timing separated", "'documented cause'", "documented cause" in flat),
    ("3.1", "calibration unit is the synthetic case", "'256 synthetic cases'",
     "256 synthetic cases" in flat),
    ("3.1", "quantile and coverage described consistently", "'quantile applied to a scored fold'",
     "quantile applied to a scored fold" in flat
     and "residuals are pooled across the three folds" not in flat),
    ("3.1", "C.14 guarantee phrase gone", "'preserved by construction' absent",
     "preserved by construction" not in flat),
    ("fmt", "body set to 12 pt", "documentclass 12pt", "\\documentclass[12pt" in s),
    ("fmt", "tables use plain comparison", "no '$<$' left", "$<" not in s),
    ("abbr", "MAST expanded", "Mega Ampere Spherical Tokamak", "Mega Ampere Spherical Tokamak" in flat),
    ("abbr", "ADAS expanded", "Atomic Data and Analysis Structure",
     "Atomic Data and Analysis Structure" in flat),
    ("abbr", "JEPA expanded", "joint-embedding predictive architecture",
     "joint-embedding predictive architecture" in flat),
    ("abbr", "QP defined", "quadratic program (QP)", "quadratic program (QP)" in flat),
    ("stats", "no local file names in the paper",
     "phrase absent", ".py" not in s.split("\\begin{thebibliography}")[0] and "A1_A2_" not in s),
]

# ---- the 2026-11-13 review
ROWS += [
    ("time", "Section 6 states the 10 ms criterion", "'must complete within 10 ms at the 99th'",
     "must complete within 10~ms at the 99th percentile" in flat),
    ("time", "no 'less than one tenth' as a criterion", "phrase with a full stop absent",
     "less than one tenth of the control period." not in flat),
    ("3.1", "no 'coverage is therefore preserved'", "phrase absent",
     "coverage is therefore preserved" not in flat),
    ("3.1", "coverage transfer hedged", "'does not establish that coverage transfers'",
     "does not establish that coverage transfers" in flat),
    ("P0-3", "fallback checked and failed, stated", "'none of them passed on any of the 120 frames'",
     "none of them passed on any of the 120 frames" in flat),
    ("fmt", "six-line table numbered", "tab:sixline", "\\label{tab:sixline}" in s),
    ("fmt", "generator table numbered", "tab:indep", "\\label{tab:indep}" in s),
    ("abbr", "MHD expanded", "magnetohydrodynamic (MHD)", "magnetohydrodynamic (MHD)" in flat),
    ("abbr", "VUV and EUV expanded", "vacuum-ultraviolet (VUV)", "vacuum-ultraviolet (VUV)" in flat),
    ("abbr", "NNLS defined", "non-negative least squares (NNLS)", "non-negative least squares (NNLS)" in flat),
    ("abbr", "ADF15 and ADF11 named", "photon-emissivity coefficient files",
     "photon-emissivity" in flat and "iso-nuclear master files" in flat),
    ("abs", "carbon inversion named", "'six-line carbon inversion'", "six-line carbon inversion" in flat),
    ("abs", "STRAHL split from the library", "'nine-case STRAHL comparison'",
     "nine-case STRAHL comparison" in flat),
    ("stats", "frame log canonical", "'canonical record'", "canonical record" in flat),
]

# ---- the 2026-11-13 23:01 review
ROWS += [
    ("3.1", "introduction names the calibration unit", "'held-out synthetic cases'",
     "held-out synthetic cases" in flat),
    ("3.1", "no 'calibrated on held-out discharges'", "phrase absent",
     "calibrated on held-out discharges" not in flat),
    ("time", "no composite 3.8 ms claim", "phrase absent", "composite of the observation update" not in flat),
    ("time", "shield profiled, not bounded", "'can be profiled independently'",
     "execution latency can be profiled independently" in flat),
    ("P0-3", "clearance and certificate separated", "'tab:safetytests'", "\\label{tab:safetytests}" in s),
]

ROWS += [
    ("4", "generator sentence is a list, not a doubled colon", "'ADF15 for photon-emissivity'",
     "ADF15 for photon-emissivity coefficients" in flat and "it needs: ADF15" not in flat),
    ("4", "same-ADAS limitation in prose", "'would appear on both sides'",
     "would appear on both sides" in flat),
]

ROWS += [
    ("C", "no repository machine paths in the paper", "phrase absent",
     "tests/results" not in flat and "stage5/data" not in flat),
    ("C", "artefacts named by identifier only", "'package manifest lists every file'",
     "supplementary material" in flat and "A14\\_Preregistration" not in s),

]

# ---- the 2026-11-15 review, P0 items
ROWS += [
    ("P0-1", "title states the narrower contribution", "'identifiability-aware'", "identifiability-aware" in flat),
    ("P0-1", "intro states the contribution in one sentence", "'the contribution claimed here'",
     "the contribution claimed here" in flat and "records the decision" in flat),
    ("P0-2", "release rate called a release rate", "'which is a release rate'", "which is a release rate" in flat),
    ("P0-3", "oracle input named as such", "'ideal-input baseline'", "ideal-input baseline" in flat),
    ("P0-4", "reproducible coverage reported", "'0.901'", "0.901" in flat and "0.867 when the" in flat),
    ("P0-4", "archived coverage called optimistic", "'optimistic'", "optimistic" in flat),
    ("P0-5", "table says what a verifier cannot verify", "'does not verify'", "does not verify" in flat),
    ("P0-5", "bias row now has an injection experiment", "'with the interval width as the control'",
     "with the interval width as the control" in flat and "Controlled bias injection" in flat),
    ("P0-6", "end-to-end period not met, stated", "'above the 10 ms period'", "above the 10 ms period" in flat),
]

ROWS += [
    ("2.1", "null direction is quantitative, not asserted", "'1-\\cos\\varphi'",
     "\\cos\\varphi" in s and "\\sigma/\\sqrt{1-\\cos\\varphi}" in s),
    ("2.1", "decay rate keeps the convective term", "'Dk^{2}+Vk'", "Dk^{2}+Vk" in s),
]

ROWS += [
    ("2.2", "grades are tabulated, not asserted", "'tab:ident'", "\\input{table_ident}" in s and "\\ref{tab:ident}" in s),
    ("2.2", "the requirement is stated per column", "'the accuracy required of that target'",
     "the accuracy required of that target" in flat),
]

ROWS += [
    ("2.3", "the table is not called six inequalities", "'condition that can be checked'",
     "condition that can be checked" in flat and "every row is an inequality" not in flat),
    ("2.3", "lock claim matches A14", "'C1--C5'", "A14, C1--C5" in flat),
]

ROWS += [
    ("2.2", "grades are stable across the grid", "'2.8 and 3.2 per cent'", "2.8 and 3.2 per cent" in flat),
]

ROWS += [
    ("C", "repository address present", "'github.com/hf-jz/FR-PWM_code'",
     "github.com/hf-jz/FR-PWM_code" in flat),
    ("C", "repository address in the abstract", "'Code and archived records are at'",
     "Code and archived records are at" in flat),
]

import zipfile
_docx = zipfile.ZipFile("frpwm_pst.docx").read("word/document.xml").decode("utf-8")
_text = re.sub(r"<[^>]+>", "", _docx)
ROWS.append(("fmt", "docx carries no raw LaTeX", "0 '$' and 0 '\\psi'",
             _text.count("$") == 0 and _text.count("\\psi") == 0))

bad = [r for r in ROWS if not r[3]]
for item, what, check, ok in ROWS:
    print("%-6s %-42s %-46s %s" % (item, what, check, "PASS" if ok else "MISSING"))
print("\n%d/%d review items applied" % (len(ROWS) - len(bad), len(ROWS)))
sys.exit(1 if bad else 0)
