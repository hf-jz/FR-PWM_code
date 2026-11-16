"""FR-PWM 实时数据服务：每 0.5s 推进一步流水线，/api/state 输出，UI 轮询。
回放模式：FRPWM_REPLAY=/path/shot.zarr → MastReplayPlant 真实 MAST shot 驱动
（顺序推进真实时间序列，循环回放），载荷带 shot 字段。"""
import json, os, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ..runtime.pipeline import FrpwmMVP

PORT = 8767
_replay = os.environ.get("FRPWM_REPLAY")
if _replay and os.path.isdir(_replay):
    from ..data.mast_replay import MastReplayCarousel
    _zps = sorted(os.path.join(_replay, d) for d in os.listdir(_replay) if d.endswith(".zarr"))
    app = FrpwmMVP(plant=MastReplayCarousel(_zps))
    app.train_neural(n=60)                        # 首炮训练神经层，跨炮泛化（OOD 诚实反映差异）
elif _replay:
    from ..data.mast_replay import MastReplayPlant
    app = FrpwmMVP(plant=MastReplayPlant(_replay))
    app.train_neural(n=60)
else:
    app = FrpwmMVP()
_hist = {"times": [], "zeff": [], "c_est": [], "Q": [], "latency": [],
         "decision": [], "odd_beta": [], "odd_q95": [], "odd_rad": [],
         "sig_meas": [], "sig_cal": [], "sig_epi": [], "sig_mf": [], "sig_proc": [],
         "dz": [], "lag": [], "risk_p": [], "brem": [], "c_src": [], "rc": {}}
from collections import deque
_rc_frames = deque(maxlen=120)                         # 逐帧 reason_code（滑窗计数源）
_latest = {}
_events = []                                        # 真实事件日志（转换触发，非周期轮询）
_prev = {}
_BUDGET = None
try:
    _BUDGET = json.load(open(os.path.join(os.path.dirname(__file__),
                                          "..", "..", "freeze", "budget_verified.json")))
except Exception:  # noqa: BLE001
    _BUDGET = None


def _step():
    s = app.step()
    h = _hist
    h["times"].append(round(s["t"], 3))
    for k in ("zeff", ):
        h[k].append(s["zeff_los"])
    h["c_est"].append(s["c_est"]); h["Q"].append(s["Q"]); h["latency"].append(s["latency_ms"])
    # 安全/ODD/Σ 五源/双空间/慢态滞后 历史（三新页数据源）
    dec_code = {"approved": 0, "corrected": 1, "rejected": 2}.get(
        (s.get("decision") or {}).get("status"), -1)
    h["decision"].append(dec_code)
    od = s.get("odd_state") or {}
    h["odd_beta"].append(od.get("beta_N")); h["odd_q95"].append(od.get("q95"))
    h["odd_rad"].append(od.get("rad_frac"))
    sg = s.get("sigma") or {}
    for k in ("meas", "cal", "epi", "mf", "proc"):
        h["sig_" + k].append(sg.get(k))
    h["dz"].append(s.get("dz"))
    al = s.get("aligned") or {}
    h["lag"].append(None if al.get("t_slow_src") is None else s["t"] - al["t_slow_src"])
    h["risk_p"].append(max((r["P"] for r in s.get("risks") or []), default=0.0))
    h["brem"].append(s.get("brem"))
    h["c_src"].append(s.get("c_src"))
    # reason_code 滑窗计数（安全裁决页 c-safe-rc 数据源：真实 120 帧滑窗，非无限累计）
    _rc_frames.append(tuple((s.get("decision") or {}).get("reason_code") or ()))
    from collections import Counter
    rc_cnt = Counter(c for fr_ in _rc_frames for c in fr_)
    h["rc"] = {k: v for k, v in rc_cnt.items()}
    for k in h:
        if k == "rc":
            continue
        h[k] = h[k][-120:]
    # 事件日志：仅在状态转换时记录（decision/degrade/shot/模块故障）
    def _ev(key, val, msg, cls=""):
        if _prev.get(key) != val:
            _events.append({"t": round(s["t"], 3), "msg": msg, "cls": cls})
            _prev[key] = val
    _ev("decision", (s.get("decision") or {}).get("status"),
        f"SafetyDecision → {(s.get('decision') or {}).get('status')}"
        + (f" ({';'.join((s.get('decision') or {}).get('reason_code') or ())})"
           if (s.get("decision") or {}).get("reason_code") else ""),
        "warn-c" if (s.get("decision") or {}).get("status") == "rejected" else "ok")
    _ev("degrade", s.get("degrade"), f"降级级 → L{s.get('degrade')}",
        "warn-c" if s.get("degrade") else "ok")
    for m, hh in (s.get("health") or {}).items():
        if hh != "ok":
            _ev("fault_" + m, hh, f"模块 {m} 故障: {hh}", "warn-c")
    shot_now = getattr(app.plant, "shot_id", None)
    if shot_now is not None:
        _ev("shot", shot_now, f"真实 Shot #{shot_now} 回放开始", "ok")
    _events[:] = _events[-20:]
    s2 = dict(s); s2["hist"] = h
    s2["ledger"] = app.ledger.report()
    if _BUDGET is not None:
        s2["budget"] = _BUDGET
    shot = getattr(app.plant, "shot_id", None)
    if shot is not None:
        s2["shot"] = shot
    done = getattr(app.plant, "shots_done", None)
    if done is not None:
        s2["shots_done"] = done
        s2["carousel_n"] = len(app.plant.plants)
    s2["events"] = list(reversed(_events[-12:]))
    globals()["_latest"] = s2


def _run():
    while True:
        try:
            _step()
        except Exception as e:  # noqa: BLE001
            print("step fault:", e)
        time.sleep(0.5)


class H(BaseHTTPRequestHandler):
    def _send(self, obj):
        b = json.dumps(obj, default=float).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/state"):
            return self._send(_latest or {"boot": True})
        if self.path.startswith("/api/demo"):
            p = os.path.join(os.path.dirname(__file__), "..", "..",
                             "tests", "results", "demo_mast_report.json")
            try:
                return self._send(json.load(open(p)))
            except Exception:  # noqa: BLE001
                return self._send({"empty": True})
        self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass


def main():
    threading.Thread(target=_run, daemon=True).start()
    print(f"frpwm server: http://127.0.0.1:{PORT}/api/state")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

if __name__ == "__main__":
    main()
