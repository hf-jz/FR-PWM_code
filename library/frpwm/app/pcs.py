"""PCS 接口层（P2-11，报告附录 C：执行器约束 + 帧序号 + watchdog 心跳联锁）。
Stage C/D 的落地接口：AI u_safe 帧 → PCS 模拟器 → 执行器指令。"""
import time
import numpy as np

class PcsEmulator:
    ACTUATORS = {  # 量程 lo,hi, slew/帧, FallbackPolicy（2.1.3 FVP：丢帧按执行器分级，禁统一 HoldLast）
        "heat": (-1.0, 1.0, 0.2, "Hold"),          # 加热惯性量 → 保持
        "gas": (0.0, 1.0, 0.2, "RampToSafe"),      # 燃料 → slew 限速撤到安全值 0
        "nbi": (0.0, 8.0, 0.1, "Zero")}            # 快响应/失效即危险 → 立即零
    POLICIES = ("Hold", "RampToSafe", "Zero", "SafePreset")
    SAFE_PRESET = {"heat": 0.0, "gas": 0.0, "nbi": 0.0}
    def __init__(self, heartbeat_miss_max=3):
        self.u = {k: 0.0 for k in self.ACTUATORS}
        self.frame_id = 0
        self.last_hb = time.time()
        self.hmm = heartbeat_miss_max
        self.interlock = False
        self.log = []
    def heartbeat(self):
        self.last_hb = time.time(); self.interlock = False
    def watchdog(self):
        """主循环周期检查：心跳缺失 >N 拍 → 联锁回退传统控制。"""
        if (time.time() - self.last_hb) > 0.005 * self.hmm:
            self.interlock = True
        return self.interlock
    def _fallback(self):
        """丢帧/联锁处置：按执行器策略分级（未登记执行器默认 SafePreset，2.1.3 冻结）。"""
        for k, spec in self.ACTUATORS.items():
            lo, hi, slew, pol = spec
            tgt = self.SAFE_PRESET.get(k, 0.0)
            if pol == "Hold":
                continue                                   # 保持上一帧
            elif pol == "RampToSafe":
                step = min(slew, abs(self.u[k] - tgt))
                self.u[k] += (1 if tgt > self.u[k] else -1) * step
            elif pol == "Zero":
                self.u[k] = 0.0
            else:                                          # SafePreset（默认档）
                self.u[k] = tgt
            self.u[k] = float(np.clip(self.u[k], lo, hi))
    def apply(self, u_cmd: dict, frame_id=None, provenance=None):
        """接收 AI 帧：帧序号检查 + slew/range 钳制；联锁态拒绝 AI 帧走按执行器分级安全回退。"""
        if self.watchdog():
            self._fallback()
            self.log.append(("interlock_reject", frame_id, dict(self.u)))
            return dict(self.u), False
        if frame_id is not None and frame_id <= self.frame_id:
            self.log.append(("stale_frame_reject", frame_id))
            return dict(self.u), False
        if frame_id is not None:
            self.frame_id = frame_id
        out = {}
        for k, (lo, hi, slew, _pol) in self.ACTUATORS.items():
            if k not in u_cmd:
                continue
            v = float(np.clip(u_cmd[k], self.u[k] - slew, self.u[k] + slew))
            self.u[k] = float(np.clip(v, lo, hi))
            out[k] = self.u[k]
        self.log.append(("apply", self.frame_id, dict(self.u)))
        return dict(self.u), True

if __name__ == "__main__":
    pcs = PcsEmulator()
    u, ok = pcs.apply({"heat": 0.9}, frame_id=1)
    assert ok and abs(u["heat"] - 0.2) < 1e-9          # slew 0.2 钳制
    u, ok = pcs.apply({"heat": 0.0}, frame_id=1)       # 重复帧号拒绝
    assert not ok
    u, ok = pcs.apply({"heat": 0.0}, frame_id=2)
    assert ok and abs(u["heat"] - 0.0) < 1e-9
    time.sleep(0.02)
    u, ok = pcs.apply({"heat": 1.0}, frame_id=3)       # 心跳超时 → 联锁
    assert not ok and pcs.interlock
    pcs.heartbeat()
    u, ok = pcs.apply({"heat": 0.3}, frame_id=4)
    assert ok
    # 2.1.3 FVP：丢帧按执行器分级回退
    pcs2 = PcsEmulator()
    pcs2.apply({"heat": 0.6, "gas": 0.9, "nbi": 5.0}, frame_id=1)
    time.sleep(0.02)
    u, ok = pcs2.apply({"heat": 0.0}, frame_id=2)       # 联锁 → 分级回退
    assert not ok
    assert abs(u["heat"] - 0.2) < 1e-9                  # heat=Hold 保持（帧1 slew 后 0.2）
    assert abs(u["gas"] - 0.0) < 1e-9                   # gas=RampToSafe slew 0.2 撤到安全值
    assert u["nbi"] == 0.0                              # nbi=Zero 立即零
    print("pcs OK: slew 钳制/帧序号/联锁回退全部生效, log:", len(pcs.log))
