"""FEC-08 统一系统级控制状态机：AUTONOMOUS → DEGRADED → SAFE_HOLD → SAFE_PCS。
恢复方向仅 SAFE_PCS → DEGRADED → AUTONOMOUS，且须连续 Nrecover 个有效周期 + 安全条件复核。
多源条件（ODD/Qt/OOD/deadline/SafetySupport）统一裁决，禁止各处不一致实现。"""
from .. import constants

STATES = ("AUTONOMOUS", "DEGRADED", "SAFE_HOLD", "SAFE_PCS")

class SafetyStateMachine:
    """每拍输入条件字典 → 输出 (state, reason)。恢复门全量满足才升级。"""
    def __init__(self, n_recover=None):
        self.state = "DEGRADED"                 # 冷启动不从 AUTONOMOUS 起（安全默认）
        self.n_recover = constants.get("conf_n_min", 200) if n_recover is None else n_recover
        self._up_streak = 0
    def _demote_target(self, cond):
        """降级目标（取最严重）。"""
        if (cond.get("lookup_miss") or cond.get("watchdog_fail") or cond.get("physics_only_unavail")
                or cond.get("input_missing") or cond.get("revision_mismatch")):
            return "SAFE_PCS", "critical:fallback_unavailable"
        if cond.get("qp_invalid") or cond.get("local_validity_fail") or \
           (cond.get("prediction_stale") and cond.get("physics_only_avail", True)):
            return "SAFE_HOLD", "hold:qp_or_validity"
        if (cond.get("safety_support") == "NONE"
                or (cond.get("safety_support") == "CONDITIONAL" and cond.get("ident_weak"))):
            return "SAFE_HOLD", "hold:support_none_or_gate_fail"   # 2.2：NONE/CONDITIONAL门不满足→Autonomous=false（原漏NONE直坠AUTONOMOUS）
        if (cond.get("odd_stale") or cond.get("ood") or cond.get("q_low")
                or cond.get("safety_support") == "PARTIAL" or cond.get("ident_weak")):
            return "DEGRADED", "degrade:conditions"
        return "AUTONOMOUS", ""
    def _recovery_ok(self, cond):
        sup = cond.get("safety_support")
        return (cond.get("odd_valid", False) and cond.get("fresh_valid", False)
                and (sup == "FULL" or (sup == "CONDITIONAL" and cond.get("ident_ok", False)))
                and cond.get("ident_ok", False)
                and cond.get("kernel_selftest", False) and not cond.get("active_fault", False))
    def step(self, cond):
        order = {s: i for i, s in enumerate(STATES)}
        tgt, reason = self._demote_target(cond)
        if order[tgt] > order[self.state]:              # 降级立即生效
            self.state, self._up_streak = tgt, 0
            return self.state, reason
        if order[tgt] == order[self.state]:
            self._up_streak = 0
            return self.state, reason or "hold"
        # 恢复：条件全部满足 → 连续 Nrecover 拍后逐级回升
        if self._recovery_ok(cond) and tgt == "AUTONOMOUS":
            self._up_streak += 1
            if self._up_streak >= self.n_recover:
                self.state = STATES[order[self.state] - 1]
                self._up_streak = 0
                return self.state, "recover:step_up"
            return self.state, f"recovering:{self._up_streak}/{self.n_recover}"
        self._up_streak = 0
        return self.state, "recover:gate_blocked"

if __name__ == "__main__":
    sm = SafetyStateMachine(n_recover=3)
    good = dict(odd_valid=True, fresh_valid=True, safety_support="FULL",
                ident_ok=True, kernel_selftest=True)
    assert sm.step(good)[0] == "DEGRADED"                   # 冷启动 DEGRADED，恢复需 streak
    assert sm.step(good)[0] == "DEGRADED"
    assert sm.step(good)[0] == "AUTONOMOUS"                 # 3 连拍 → 升级
    assert sm.step(dict(odd_stale=True))[0] == "DEGRADED"   # 降级立即
    assert sm.step(dict(qp_invalid=True))[0] == "SAFE_HOLD"
    assert sm.step(dict(watchdog_fail=True))[0] == "SAFE_PCS"
    assert [sm.step(good)[0] for _ in range(3)][-1] == "SAFE_HOLD"   # 逐级回升
    assert [sm.step(good)[0] for _ in range(3)][-1] == "DEGRADED"
    assert [sm.step(good)[0] for _ in range(3)][-1] == "AUTONOMOUS"
    assert sm.step(dict(ood=True))[0] == "DEGRADED"         # 恢复中插入降级 → streak 清零
    print("state_machine OK: 降级立即/恢复 Nrecover 逐级/门阻塞全过")
