









# FR-PWM: code and archived records

Code and the A1--A15 records behind every quantitative claim in the manuscript.

| folder | content |
|---|---|
| `figures/` | one script per figure, plus the identifiability table and the temperature sweep |
| `analysis/` | identifiability, ablation, contraction, timing, QP, conformal and bias-injection studies |
| `tools/` | package assembly, number verification and the wording audit used in the build |
| `library/` | the modelling library the scripts import |
| `artefacts/` | the A1--A15 records, a SHA-256 manifest and the recomputed summaries |

## Check a number

```bash
python3 tools/check_manifest.py
```

It verifies every file under `artefacts/` against the digest recorded in `artefacts/manifest.json`.
Table 7 of the paper names the record (A1--A15) behind each figure and table.

## Regenerate

Each figure script reads its record from `artefacts/` and asserts the values it plots, so a mismatch stops
it instead of producing a plausible figure. `tools/verify_numbers.py` is the gate used in the build: it
rechecks every number printed in the paper against these records.
