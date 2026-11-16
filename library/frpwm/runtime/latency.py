"""延迟台账（P1-7，报告 17/18：D_i 模块 deadline vs D_system 系统 critical path，
P50/P99/P999/WCET 统计）。WCET = 观测到的最大执行时间（硬实时安全指标）。"""
import time
import numpy as np

class LatencyLedger:
    BUDGET = {"fabric": 1.0, "physics_chain": 100.0, "jepa": 2.0, "hrm": 10.0,
              "trm": 10.0, "worldmodel": 10.0, "risk": 5.0, "safety": 1.0,
              "supervisor": 1.0, "control": 10.0}   # ms，模块 deadline D_i
    D_SYSTEM = 10.0                                    # critical path P99 预算 ms
    SAFETY_RESERVATION = 1.5                           # FEC-07：Safety Shield+PCS handoff 硬时间份额（MPC 不可占用）
    def __init__(self):
        self.rec = {}
    def tick(self, name, fn):
        t0 = time.perf_counter()
        r = fn()
        ms = (time.perf_counter() - t0) * 1000
        self.rec.setdefault(name, []).append(ms)
        return r
    def stats(self, name):
        a = np.array(self.rec.get(name, [0.0]))
        q = lambda p: float(np.percentile(a, p))
        return dict(n=len(a), P50=q(50), P99=q(99), P999=q(99.9), WCET=float(a.max()),
                    budget=self.BUDGET.get(name))
    def report(self):
        rep = {k: self.stats(k) for k in self.rec}
        # 系统 critical path = 快环模块之和（慢环并行不计入）
        fast = ["fabric", "jepa", "hrm", "risk", "safety", "control"]
        n = max((len(self.rec.get(m, [])) for m in fast), default=0)
        crit = [sum(self.rec.get(m, [0.0])[i] if i < len(self.rec.get(m, [])) else 0.0
                    for m in fast) for i in range(n)]
        a = np.array(crit) if crit else np.array([0.0])
        p50 = float(np.percentile(a, 50))
        jit = np.abs(a - p50)
        rep["system"] = dict(P99=float(np.percentile(a, 99)) if crit else 0.0,
                             P999=float(np.percentile(a, 99.9)) if crit else 0.0,
                             WCET=float(max(crit)) if crit else 0.0,
                             jitter_P999=float(np.percentile(jit, 99.9)) if crit else 0.0,
                             deadline_miss_rate=float((a > self.D_SYSTEM).mean()) if crit else 0.0,
                             budget=self.D_SYSTEM,
                             safety_reservation_ms=self.SAFETY_RESERVATION)
        return rep
    def misses(self):
        """D_i 违约清单（WCET 超预算的模块）+ D_system 违约。"""
        rep = self.report()
        out = [k for k, v in rep.items()
               if k != "system" and v.get("budget") and v["WCET"] > v["budget"]]
        if rep["system"]["P99"] > self.D_SYSTEM:
            out.append("system")
        return out

if __name__ == "__main__":
    led = LatencyLedger()
    for _ in range(50):
        led.tick("safety", lambda: None)
        led.tick("jepa", lambda: time.sleep(0.0002))
    rep = led.report()
    assert rep["jepa"]["WCET"] >= rep["jepa"]["P999"] >= rep["jepa"]["P99"]
    assert "system" in rep and not led.misses()
    print("latency OK:", {k: {kk: round(vv, 3) for kk, vv in v.items() if kk != "n"}
                          for k, v in rep.items()})
