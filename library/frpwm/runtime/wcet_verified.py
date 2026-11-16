"""B^verified：HIL 实测 WCET 替换设计预算（报告 2.1.2 ③）。
B^design=附录 A 设计预算表；B^verified=实测 WCET<T_deadline 逐段达成后升级，
产出 freeze/budget_verified.json 台账。未升级前不得声明"实时性已证明"。"""
import json, os, time
import numpy as np

# 附录 A Fast Critical Path 设计预算（P99/WCET, ms）
BUDGET_DESIGN = {
    "capture":   (0.3, 0.5), "transport": (0.4, 0.7), "preprocess": (0.5, 0.8),
    "jepa":      (1.0, 1.5), "observer":  (0.5, 0.8), "mpc":        (3.0, 3.2),
    "safety":    (0.7, 0.9), "ipc":       (0.5, 0.8), "margin":     (1.1, 0.0),
}
T_DEADLINE_MS = 10.0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "freeze", "budget_verified.json")   # 与 freeze/manifest.json 同目录

def measure_fast_path(n=300, seed=0, warmup=3):
    """在合成 HIL 链上实测快环各段（perf_counter，ms）→ {seg: (P99, WCET)}。
    冷启动（首拍 PecCache 建表/ADAS I/O）按架构纪律属离线 init，单独记录不计入在线 WCET。"""
    from .pipeline import FrpwmMVP, SyntheticPlant
    from ..app.pcs import PcsEmulator
    app = FrpwmMVP(plant=SyntheticPlant(seed=seed))
    t0 = time.perf_counter()
    for i in range(warmup):                      # 冷启动：init 预建表（离线语义）
        app.step(t=-1.0 + i * 0.001)
    cold_ms = (time.perf_counter() - t0) * 1000
    pcs = PcsEmulator()
    seg = {k: [] for k in BUDGET_DESIGN}
    slow_ms = []
    def tick(name, fn):
        t0 = time.perf_counter(); r = fn()
        seg[name].append((time.perf_counter() - t0) * 1000)
        return r
    for i in range(n):
        t_f = 20.0 * i / n
        fr = tick("capture", lambda: app.plant.frame(t=t_f))
        tick("transport", lambda: [app.fabric.ingest("spec", fr["Te"]),
                                   app.fabric.ingest("thomson", fr["ne"])])
        tick("preprocess", lambda: np.concatenate([fr["Te"][:5], fr["ne"][:5] * 1e-19]))
        t0s = time.perf_counter()                # 物理正演链=慢环（不在快环 critical path）
        phys = app._mod("physics_chain", lambda: app._physics_chain(fr), None)
        slow_ms.append((time.perf_counter() - t0s) * 1000)
        if app.jepa is not None:
            tick("jepa", lambda: app.jepa.encode(fr["Te"][::12][None, :]))
        else:
            seg["jepa"].append(0.0)
        tick("observer", lambda: app._mod("ood", lambda: app.ood.score(
            np.concatenate([fr["Te"][:5], fr["ne"][:5] * 1e-19]))[0] if app.ood else 0.0, 0.0))
        risks = app.risk.infer(dict(mhd_amp=fr["mhd_amp"], d2e_dt=fr["d2e_dt"],
                                    rad_frac=fr["rad_frac"], beta_N=fr["beta_N"], ood=0.0), {})
        if app.mpc is not None and app.control_enabled:
            rk = (float(risks[0].P), float(risks[0].tte)) if risks else None
            tick("mpc", lambda: app.mpc.act(fr["Te"], 3050.0, risk=rk))
        else:
            seg["mpc"].append(0.0)
        tick("safety", lambda: app.shield.check(
            dict(beta_N=fr["beta_N"], q95=fr["q95"],
                 zeff=phys["zeff_los"] if phys else 2.0, rad_frac=fr["rad_frac"]),
            app.fabric.stale_channels(), 0.0))
        tick("ipc", lambda: pcs.apply({"heat": 0.0}, frame_id=i + 1))
        seg["margin"].append(0.0)
    out = {}
    for k, v in seg.items():
        a = np.array(v or [0.0])
        out[k] = (float(np.percentile(a, 99)), float(a.max()))
    out["_cold_start_ms"] = round(cold_ms, 1)    # 冷启动单列（离线 init，不在在线预算）
    sa = np.array(slow_ms or [0.0])
    out["_slow_path_ms"] = dict(p99=float(np.percentile(sa, 99)), wcet=float(sa.max()))
    return out

def verify(measured, path=OUT):
    """逐段对照设计 WCET + 合计 < T_deadline → B^verified 台账。
    慢环（物理正演/TRM）独立按 T_slow,deadline=100ms 验收，不计入快环合计。"""
    cold = measured.pop("_cold_start_ms", None)
    slow = measured.pop("_slow_path_ms", None)
    rows, ok_all, total = {}, True, 0.0
    for k, (p99_d, wcet_d) in BUDGET_DESIGN.items():
        p99_m, wcet_m = measured[k]
        ok = wcet_m <= wcet_d
        ok_all &= ok
        total += wcet_m
        rows[k] = dict(design_p99=p99_d, design_wcet=wcet_d,
                       measured_p99=round(p99_m, 3), measured_wcet=round(wcet_m, 3), pass_=ok)
    rows["total"] = dict(design_wcet=9.2, measured_wcet=round(total, 3),
                         pass_=total < T_DEADLINE_MS)
    ok_all &= total < T_DEADLINE_MS
    payload = dict(status="B^verified" if ok_all else "B^design(unverified)",
                   deadline_ms=T_DEADLINE_MS, cold_start_ms=cold,
                   cold_note="冷启动=离线 init（PecCache 建表/ADAS I/O），按纪律不进在线环",
                   slow_path=dict(budget_ms=100.0, measured=slow,
                                  pass_=bool(slow and slow["wcet"] < 100.0)),
                   segments=rows,
                   note="仅 B^verified 可声明实时性已证明" if ok_all
                        else "存在超设计段，保持 B^design，禁止声明实时性已证明")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

if __name__ == "__main__":
    m = measure_fast_path(300)
    r = verify(m)
    for k, v in r["segments"].items():
        print(f"  {k:10s} design_wcet={v['design_wcet']:>4}  measured_wcet={v['measured_wcet']:>7}  {'OK' if v['pass_'] else 'OVER'}")
    print("wcet_verified:", r["status"])
