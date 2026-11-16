"""Path resolution: records ship with the repository, external data comes from the environment."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTEFACTS = os.path.join(ROOT, "artefacts")
LIBRARY = os.path.join(ROOT, "library")
OUT = os.environ.get("FRPWM_OUT", os.path.join(ROOT, "generated"))

_RECORD_MAP = {
    "tests/results/upsse/report.json": "A1_A2_Closure_Calibration/upsse/report.json",
    "tests/results/pgm_zeff/report.json": "A1_A2_Closure_Calibration/pgm_zeff/report.json",
    "tests/results/inversion/inversion_report.json": "A1_A2_Closure_Calibration/inversion/inversion_report.json",
    "tests/results/strahl_pinn/strahl_pinn_report.json": "A1_A2_Closure_Calibration/strahl_pinn/strahl_pinn_report.json",
    "tests/results/demo_mast_report.json": "A5_MAST_60Shot_Replay/demo_mast_report.json",
    "tests/results/t3e4_rollout.json": "A11_Rollout/t3e4_rollout.json",
    "tests/results/ident4/ident4_report.json": "A12_Fisher_Observation_Sets/ident4_report.json",
    "tests/results/safety_wcet.json": "A9_A10_Timing/safety_wcet.json",
    "tests/results/wcet_upsse/report.json": "A9_A10_Timing/wcet_upsse/report.json",
    "tests/results/ablation": "A8_Diagnostic_Ablation/ablation",
    "tests/results/ablation/base_folds.json": "A8_Diagnostic_Ablation/ablation/base_folds.json",
    "tests/results/ident4/ident4_report.json": "A12_Fisher_Observation_Sets/ident4_report.json",
    "tests/results/upsse/report.json": "A1_A2_Closure_Calibration/upsse/report.json",
    "A6_Uncertainty_Budget/uncertainty_budget.csv": "A6_Uncertainty_Budget/uncertainty_budget.csv",
    "A1_A2_Closure_Calibration/conformal_rows.csv": "A1_A2_Closure_Calibration/conformal_rows.csv",
    "A5_MAST_60Shot_Replay/mast_mast_frame_logs.csv": "A5_MAST_60Shot_Replay/mast_mast_frame_logs.csv",
}


def record(*parts):
    """Absolute path of an archived record, resolved through the map above."""
    rel = "/".join(parts)
    return os.path.join(ARTEFACTS, _RECORD_MAP.get(rel, rel))


def external(name, env, hint):
    """Data that cannot be shipped; refuse loudly rather than guess at a home directory."""
    value = os.environ.get(env)
    if not value:
        raise SystemExit("%s is not set: %s" % (env, hint))
    return os.path.join(value, name)
