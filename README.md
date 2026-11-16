









# FR-PWM: code and archived records

Code and the A1--A16 records behind every quantitative claim in the manuscript.

| folder | content |
|---|---|
| `figures/` | one script per figure, plus the identifiability table and the temperature sweep |
| `analysis/` | identifiability, ablation, contraction, timing, QP, conformal, bias-injection and spectral-provenance studies |
| `tools/` | the package check and the manifest verifier a reader can run without the paper source |
| `library/` | the modelling library the scripts import |
| `artefacts/` | the A1--A15 records, a SHA-256 manifest and the recomputed summaries |

Released under the MIT licence; see `LICENSE`.

## What runs with nothing installed

Measured on a fresh clone with every external-data variable unset: **19 of the 23 scripts run to
completion**, and `tools/check_manifest.py` reports 66 files checked, 0 problems afterwards, so the
examples do not overwrite the records they read. The four that do not run need one of these:

| script | why |
|---|---|
| `analysis/ident_4combos.py` | rebuilds the transport forward model through `strahl-cpp` (`STRAHL_CPP`) |
| `figures/stage30_grid_crlb.py` | rebuilds the observation operator from ADAS data (`ADAS_FULL`, `SPEC2CONC`) |
| `analysis/wcet_upsse.py` | runs, but its own latency gate is machine-dependent and fails on a laptop |
| `analysis/run_ablation.py` | runs, but only after `diag_ablate.py` has produced its intermediates |

Each of them stops with a sentence naming the variable to set; none of them silently substitutes data.


## Dependencies

Pure Python (numpy, scipy, matplotlib, pandas). Everything that reads an archived record runs with no
other software; regenerating the records needs the external data named below.

| what you want to run | needs |
|---|---|
| `figures/` | numpy, matplotlib, and the records in `artefacts/` |
| `analysis/` | the modelling library in `library/` |
| records A1--A4, A12 | ADAS data (`ADAS_FULL`) and the ionisation-balance code (`SPEC2CONC`) |
| records A2, A6, A7 | a compiled `strahl-cpp` (`STRAHL_CPP`) and the STRAHL atomic data (`STRAHL_ROOT`) |
| record A5, real-spectra checks | the public MAST archive (`TOKAMARK_DIR`) and EFIT geometry (`MAST_GILE`) |

JOREK is cited for a coordinate check that contributes no result; no code here calls it.

## Paths

Nothing in this repository refers to a path outside it. `paths.py` resolves archived records under
`artefacts/` and writes generated output to `$FRPWM_OUT` (default `generated/`). External data that cannot
be redistributed is read from an environment variable and the script stops with a message naming it:
`ADAS_FULL`, `SPEC2CONC`, `STRAHL_ROOT`, `STRAHL_CPP`, `MAST_GILE`, `TOKAMARK_DIR`.

## Check a number

```bash
python3 tools/check_manifest.py
```

It verifies every file under `artefacts/` against the digest recorded in `artefacts/manifest.json`.
Table 7 of the paper names the record (A1--A15) behind each figure and table.

## Regenerate

Each figure script reads its record from `artefacts/` and asserts the values it plots, so a mismatch stops
it instead of producing a plausible figure. `tools/verify_numbers.py` is the gate used in the build: it
is part of the paper's build tree and is not shipped here; `tools/check_manifest.py` is the self-contained check.
