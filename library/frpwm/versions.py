"""版本 ID + Replay 逐位复现（P1-6）。
2.1.2 收口：BundleID 唯一事实源为 freeze_manifest.MANIFEST（8 字段 canonical SHA-256）；
本模块 VersionIds 为其兼容别名（避免双版本系统漂移），fingerprint 供逐位复现核对。"""
import hashlib
from dataclasses import dataclass, asdict
from .freeze_manifest import MANIFEST

@dataclass(frozen=True)
class VersionIds:
    """兼容别名：字段与 BundleID 一一对应，digest() == MANIFEST.bundle_id()。"""
    model: str = MANIFEST.model
    training_data: str = MANIFEST.training_data
    physics: str = MANIFEST.physics
    atomic_db: str = MANIFEST.atomic_db
    efit: str = MANIFEST.efit
    calibration: str = MANIFEST.calibration
    device_config: str = MANIFEST.device_config
    runtime_code: str = MANIFEST.runtime_code
    def digest(self):
        return MANIFEST.bundle_id()

VERSIONS = VersionIds()

def stamp(d: dict):
    """给输出 dict 盖版本束与 BundleID（provenance version 字段同源）。"""
    d = dict(d)
    d["versions"] = asdict(VERSIONS)
    d["version_digest"] = VERSIONS.digest()
    return d

def fingerprint(state: dict, keys=("c_est", "zeff_los", "Q", "degrade")):
    """输出指纹：逐位复现核对用（浮点按 repr 固化）。"""
    s = "|".join(repr(state.get(k)) for k in keys)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

if __name__ == "__main__":
    a = stamp({"x": 1})
    assert a["version_digest"] == MANIFEST.bundle_id()
    f1 = fingerprint({"c_est": 0.03, "zeff_los": 1.17, "Q": 0.6, "degrade": 0})
    f2 = fingerprint({"c_est": 0.03, "zeff_los": 1.17, "Q": 0.6, "degrade": 0})
    f3 = fingerprint({"c_est": 0.031, "zeff_los": 1.17, "Q": 0.6, "degrade": 0})
    assert f1 == f2 and f1 != f3
    print("versions OK: BundleID 单一事实源", VERSIONS.digest())
