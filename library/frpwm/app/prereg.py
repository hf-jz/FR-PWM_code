"""预注册验收（P2-16，报告 18 章首 50 炮预注册标准：配置哈希锁定 + 逐条评估 + 不可事后改）。"""
import hashlib, json
import numpy as np

PREREG = dict(
    name="FR-PWM MVP-1 首50炮预注册验收",
    criteria=[
        dict(id="C1", metric="zeff_rel_err_p50", op="<=", thr=0.15, desc="Z_eff^LOS 相对误差 P50 ≤15%"),
        dict(id="C2", metric="latency_p99_ms", op="<=", thr=10.0, desc="快环延迟 P99 ≤10ms"),
        dict(id="C3", metric="missing_rate", op="<=", thr=0.20, desc="缺失率(不静默插值) ≤20%"),
        dict(id="C4", metric="no_control", op="==", thr=True, desc="全程不控制(shadow only)"),
        dict(id="C5", metric="stale_events", op=">=", thr=0, desc="Stale 事件全部记录"),
    ])

# 统计能力分阶段（2.1.2：50 炮仅 MVP-1 Entry Gate，不作最终统计证明；
# 逐阶段重新预注册并收紧 PICP 下界 ε_stat 与精度阈）
STAGES = {
    50: dict(role="MVP-1 Entry Gate", eps_stat=0.10, zeff_rel_err_p50=0.15),
    200: dict(role="Shadow Consolidation", eps_stat=0.06, zeff_rel_err_p50=0.12),
    500: dict(role="Pre-Production", eps_stat=0.04, zeff_rel_err_p50=0.10),
    1000: dict(role="Production Gate", eps_stat=0.03, zeff_rel_err_p50=0.08),
}

def stage_config(n_shots, base=PREREG):
    """按阶段生成预注册配置：收紧 C1 精度阈，并加 C6 PICP 单侧覆盖条。"""
    st = STAGES.get(n_shots)
    assert st is not None, f"未定义阶段 {n_shots}（合法: {sorted(STAGES)}）"
    crit = [dict(c, thr=st["zeff_rel_err_p50"]) if c["id"] == "C1" else dict(c)
            for c in base["criteria"]]
    crit.append(dict(id="C6", metric="picp90", op=">=", thr=0.90 - st["eps_stat"],
                     desc=f"PICP90 ≥ 0.90−ε_stat(={st['eps_stat']}, Clopper-Pearson 单侧)"))
    return dict(name=f"{base['name']}·{n_shots}炮[{st['role']}]", criteria=crit)

def evaluate_stage(metrics: dict, n_shots: int, locked_hash=None):
    cfg = stage_config(n_shots)
    return evaluate(metrics, cfg=cfg, locked_hash=locked_hash if locked_hash else lock(cfg))

def lock(cfg=PREREG):
    """配置哈希：注册后评估必须用同一哈希，防事后改阈值。"""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

def evaluate(metrics: dict, cfg=PREREG, locked_hash=None):
    if locked_hash is not None and lock(cfg) != locked_hash:
        return dict(valid=False, reason="预注册配置哈希不匹配（疑似事后修改）")
    res = []
    for c in cfg["criteria"]:
        v = metrics.get(c["metric"])
        ok = (v is not None) and ((v <= c["thr"]) if c["op"] == "<="
                                  else (v == c["thr"] if c["op"] == "==" else v >= c["thr"]))
        res.append(dict(id=c["id"], metric=c["metric"], value=v, thr=c["thr"], pass_=ok, desc=c["desc"]))
    return dict(valid=True, all_pass=all(r["pass_"] for r in res), items=res)

if __name__ == "__main__":
    h = lock()
    good = evaluate(dict(zeff_rel_err_p50=0.12, latency_p99_ms=6.5, missing_rate=0.05,
                         no_control=True, stale_events=3), locked_hash=h)
    assert good["valid"] and good["all_pass"]
    bad = evaluate(dict(zeff_rel_err_p50=0.30, latency_p99_ms=6.5, missing_rate=0.05,
                        no_control=True, stale_events=3), locked_hash=h)
    assert bad["valid"] and not bad["all_pass"]
    tampered = evaluate(dict(zeff_rel_err_p50=0.30), cfg=dict(PREREG, criteria=[
        dict(c, thr=0.5) if c["id"] == "C1" else c for c in PREREG["criteria"]]), locked_hash=h)
    assert not tampered["valid"]
    # 分阶段：50 炮松、1000 炮紧；同一指标在 50 炮过、在 1000 炮不过
    m = dict(zeff_rel_err_p50=0.13, latency_p99_ms=6.5, missing_rate=0.05,
             no_control=True, stale_events=3, picp90=0.85)
    r50 = evaluate_stage(m, 50)
    assert r50["all_pass"], r50["items"]
    r1000 = evaluate_stage(m, 1000)
    assert not r1000["all_pass"]                     # 0.13 > 0.08 且 0.85 < 0.87
    cfg200 = stage_config(200)
    assert any(c["id"] == "C6" for c in cfg200["criteria"])
    print("prereg OK: 达标/不达标/篡改检测/分阶段收紧 全部正确, hash:", h)
