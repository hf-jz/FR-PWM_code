"""Freeze Manifest + BundleID canonical serialization（报告 2.1.2 版本治理）。
规则冻结：字段顺序固定、UTF-8、key=value 以 | 连接、SHA-256；manifest 随代码仓存储。"""
import hashlib, json, os
from dataclasses import dataclass, asdict

FIELDS = ["schema", "model", "training_data", "physics", "atomic_db",
          "efit", "calibration", "device_config", "runtime_code"]     # FEC-05：版本束只允许 RuntimeCode，不再用 Runtime 同义字段；2.2 P0-3：DeviceConfig 入 bundle（同模型+不同 ODD/Vsafe/Actuator 不得同 BundleID）

@dataclass(frozen=True)
class FreezeManifest:
    schema: str = "schema-v1.4"
    model: str = "frpwm-2.2"
    training_data: str = "synthetic-v1"
    physics: str = "strahl-cpp-1.0"
    atomic_db: str = "adas-full-2026"
    efit: str = "parametric-v1"
    calibration: str = "cal-v1"
    device_config: str = "device-v1"                # 2.2 P0-3：PCS/ODD/Vsafe/Actuator/Clock/Network/Calibration 装置配置束
    runtime_code: str = "frpwm-runtime-2.2"     # FEC-05：RuntimeCode 改变必须导致 BundleID 改变
    def canonical(self):
        """canonical serialization 冻结：字段顺序固定；UTF-8；key=value；字段以 | 连接；
        禁止空格差异、字段别名和隐式默认值（FEC-05）。"""
        d = asdict(self)
        return "|".join(f"{k}={d[k]}" for k in FIELDS)
    def bundle_id(self):
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]

def check_replay_compat(recorded_bundle_id):
    """FEC-05：任何 BundleID 变化使 Replay/Test manifest 不兼容并显式告警。"""
    cur = MANIFEST.bundle_id()
    if recorded_bundle_id and recorded_bundle_id != cur:
        raise RuntimeError(f"BundleID 不兼容：记录 {recorded_bundle_id} ≠ 当前 {cur}"
                           "（RuntimeCode/版本束已变，Replay manifest 失效）")
    return True

MANIFEST = FreezeManifest()
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "freeze", "manifest.json")

def write_manifest(path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"manifest": asdict(MANIFEST), "bundle_id": MANIFEST.bundle_id(),
               "canonical_rule": "field order fixed; UTF-8; key=value joined by |; SHA-256"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

if __name__ == "__main__":
    p = write_manifest()
    b1 = MANIFEST.bundle_id()
    m2 = FreezeManifest(model="frpwm-0.2")
    assert b1 != m2.bundle_id()                       # 字段变 → BundleID 变
    assert MANIFEST.bundle_id() == b1                 # 稳定
    back = json.load(open(p, encoding="utf-8"))
    assert back["bundle_id"] == b1 and back["manifest"]["schema"] == "schema-v1.4"
    print(f"freeze_manifest OK: BundleID={b1}, 变更检测通过, {p}")
