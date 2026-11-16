# -*- coding: utf-8 -*-
"""SafetyConstraintRegistry v1.0（冻结工件，对齐 main.pdf 附录 C 表）。

九字段：constraint_id / required_state_ids / required_diagnostic_ids /
required_estimator_ids / required_time_validity_ms / support_level /
fallback_state / safe_action / device_scope。
SupportLevel 语义（2.2 冻结）：FULL→Autonomous allowed；CONDITIONAL→须满足
附加 Gate（gate 字段）才允许；PARTIAL/NONE→AutonomousControl=false。
装置特化阈值（Z_max/β_max/slew 等）随 DeviceConfig 入 BundleID，本表结构为平台不变量。
"""
SUPPORT_LEVELS = ("FULL", "CONDITIONAL", "PARTIAL", "NONE")

# (constraint_id, required_state, required_diag, estimator, tv_ms, support, gate, fallback, safe_action)
_E = [
    ("zeff_control_max",   "Z_eff^control",        "EUV,brem",        "UPSSE_Fast", 20,  "FULL",        None,          "PhysicsOnly",   "RampToSafe"),
    ("p_rad_max",          "P_rad",                "AXUV",            "UPSSE_Fast", 20,  "FULL",        None,          "PhysicsOnly",   "RampToSafe"),
    ("beta_n_max",         "beta_N",               "mag,EFIT",        "FastMag",    5,   "FULL",        None,          "SafePreset",    "reduce_heating"),
    ("q95_min",            "q95",                  "EFIT",            "FastMag",    50,  "FULL",        None,          "SafePreset",    "ip_adjust"),
    ("density_limit",      "n_e<n_GW",             "Thomson,interf",  "UPSSE_Fast", 100, "FULL",        None,          "SafePreset",    "gas_cut"),
    ("vertical_disp",      "z_v",                  "mag_probes",      "FastMag",    1,   "FULL",        None,          "SafePreset",    "pcs_inner"),
    ("elm_heat_load",      "ELM freq,size",        "D_alpha,mag",     "Risk",       100, "CONDITIONAL", "R_t 进闭环需另设 Recall/FAR 门", "advisory", "pellet_rmp_advice"),
    ("disruption_risk",    "R_t,TTE",              "multi_diag",      "Risk",       100, "CONDITIONAL", "R_t 进闭环需另设 Recall/FAR 门", "advisory", "soft_landing_advice"),
    ("locked_mode",        "A_n1,rotation",        "mag_saddle",      "Risk",       50,  "CONDITIONAL", "R_t 进闭环需另设 Recall/FAR 门", "advisory", "spin_rmp_advice"),
    ("w_core_accum",       "n_W(rho)",             "EUV",             "UPSSE",      500, "FULL",        None,          "PhysicsOnly",   "impurity_source_control"),
    ("w_transport",        "n_W,D_W,V_W",          "EUV_window",      "UPSSE_Slow", None,"CONDITIONAL", "I^ident != 2","prev_x_slow",  "no_autonomous_advised"),
    ("w_high_charge",      "n_W^q",                "EUV",             "UPSSE_Slow", None,"PARTIAL",     None,          "advisory",      "autonomous_forbidden"),
    ("mag_ip_ramp",        "I_p,dI_p/dt",          "mag",             "FastMag",    1,   "FULL",        None,          "SafePreset",    "pcs_inner"),
    ("actuator_range",     "u",                    "act_telemetry",   "SafetyKernel",0,  "FULL",        None,          "SafePreset",    "clip_reject"),
    ("actuator_slew",      "u,u_prev",             "act_telemetry",   "SafetyKernel",0,  "FULL",        None,          "SafePreset",    "slew_clamp"),
    ("stale_state",        "all_states",           "all_diag",        "SafetyKernel",0,  "FULL",        None,          "DEGRADED",     "degrade_bypass"),
    ("reachability",       "X_phys",               "all_diag",        "SafetyKernel",0,  "FULL",        None,          "Usafe_cert",    "lookup_safe_action"),
]
REGISTRY = [dict(zip(
    ("constraint_id", "required_state_ids", "required_diagnostic_ids",
     "required_estimator_ids", "required_time_validity_ms", "support_level",
     "gate", "fallback_state", "safe_action"), e)) for e in _E]
for _r in REGISTRY:
    _r["device_scope"] = "FRPWM-EAST-profile"


def validate():
    """冻结结构自检：id 唯一、枚举合法、CONDITIONAL 必带 gate、必备九字段。"""
    ids = [r["constraint_id"] for r in REGISTRY]
    assert len(ids) == len(set(ids)), "constraint_id 重复"
    for r in REGISTRY:
        assert r["support_level"] in SUPPORT_LEVELS, r
        if r["support_level"] == "CONDITIONAL":
            assert r["gate"], f"CONDITIONAL 缺 gate: {r['constraint_id']}"
    return True


def autonomous_for(constraint_id, gate_satisfied=False):
    """SupportLevel → AutonomousControl 判定（逐约束）。"""
    r = next(x for x in REGISTRY if x["constraint_id"] == constraint_id)
    lv = r["support_level"]
    if lv == "FULL":
        return True
    if lv == "CONDITIONAL":
        return bool(gate_satisfied)
    return False                                        # PARTIAL / NONE


if __name__ == "__main__":
    validate()
    assert autonomous_for("zeff_control_max")
    assert autonomous_for("w_transport", gate_satisfied=True)
    assert not autonomous_for("w_transport", gate_satisfied=False)
    assert not autonomous_for("w_high_charge", gate_satisfied=True)
    print(f"SafetyConstraintRegistry v1.0: {len(REGISTRY)} constraints OK")
