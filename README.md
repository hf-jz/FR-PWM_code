









# FR-PWM: code and archived records

Code and the A1--A22 records behind every quantitative claim in the manuscript.

| folder | content |
|---|---|
| `figures/` | one script per figure, plus the identifiability tables and the temperature sweep |
| `analysis/` | identifiability, ablation, contraction, timing, QP, conformal, bias-injection and spectral-provenance studies |
| `analysis/sdtif/` | the identifiability calculus of Section 3: Jacobian, whitened SVD, complete null space, target projection, nuisance profiling, robustness, state certificate, declared specifications and the real-data case |
| `tools/` | the package check, the manifest verifier and the manifest builder a reader can run without the paper source |
| `library/` | the modelling library the scripts import |
| `artefacts/` | the A1--A22 records, a SHA-256 manifest and the recomputed summaries |

Released under the MIT licence; see `LICENSE`.

## The identifiability calculus (record A22)

```bash
python3 analysis/sdtif_run.py          # six closed-form benchmarks, then the plasmid cases
python3 figures/make_table_sdtif.py    # Tables 5 and 6 of the paper, from the A22 record
python3 analysis/sdtif_run.py --skip-plasma   # benchmarks only, needs no external data
python3 tools/fetch_mast_level2.py            # re-fetch the ten MAST shots (needs network and `zstd`)
python3 analysis/p0_admissibility.py          # the barrier admissibility breakpoints (record A23)
```

`analysis/sdtif_run.py` asserts the seven analytic benchmarks before touching a plasma model, then writes
`artefacts/A22_SD_TIF/{benchmarks.json,cases.json,EVIDENCE.md}`, the ten `inputs/level2_*.npz` shots and the
   campaign table. `core.py` holds no physics; `plasma.py`
holds the three carbon operators, the transport case and the state map. The plasma cases need `ADAS_FULL`
and `SPEC2CONC`; without them the script stops naming the variable rather than substituting data.

## What runs with nothing installed

Measured by running every script in a clean environment with the external-data variables unset: **18 of the
34 scripts run to completion**. The other sixteen fall into named groups, and each stops with a sentence
naming what is missing rather than substituting data.

| group | scripts | what is missing |
|---|---|---|
| ADAS release | `analysis/stage53_certified_bound.py`, `analysis/stage55_determinism.py`, `analysis/stage57_closed_loop.py`, `analysis/stage59_model_loop.py`, `analysis/stage64_seven_unknowns.py`, `analysis/stage65_populated_functional.py`, `analysis/sdtif_run.py` | `ADAS_FULL` (the last one only for its plasma cases; `--skip-plasma` runs the benchmarks alone) |
| ADAS and the plasma_spec2conc package | `figures/stage30_grid_crlb.py` | `ADAS_FULL`, `SPEC2CONC` |
| strahl-cpp build | `analysis/ident_4combos.py` | `STRAHL_CPP` |
| the fairmast reader | `analysis/stage40_postlock_checks.py` | the `fairmast_zarr` module |
| network | `analysis/stage48_real_crosscheck.py`, `analysis/stage48b_width_vs_ti.py` | the public MAST store; each downloads ten discharges and exceeds a 90 s probe timeout |
| paper build tree | `figures/make_fig3.py`, `figures/make_realdata.py`, `analysis/stage56_evidence_map.py`, `analysis/stage67_coverage_by_regime.py` | the per-frame replay log and the record tree the paper is built from |
| machine-dependent gate | `analysis/wcet_upsse.py` | runs, and fails its own latency budget on a laptop, which is the honest result |

## Three counts, and what each one counts

| number | what it is | where it is checked |
|---|---|---|
| 168 | files in the archived record package (`artefacts/manifest.json`, field `files`) | not verified by the shipped tool: the record package travels as a zip beside the paper |
| 101 | files tracked in the repository manifest (`artefacts/manifest_repo.json`) | `tools/check_manifest.py`, which prints "101 files checked, 0 problems" |
| 21 of 39 | scripts that run with no external data | the table above; `analysis/sdtif_run.py --skip-plasma` needs nothing at all |

The canonical check is `python3 tools/check_manifest.py`: it verifies the repository files against their digests and exits non-zero on any mismatch. The record count is a property of the archive and the script count a property of the environment, so all three are stated rather than reconciled.



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
