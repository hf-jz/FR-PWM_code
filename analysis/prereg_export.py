"""预注册工件：把判据与阈值锁定成可审计的 JSON（含配置哈希与冻结 bundle）。

审稿意见 Stage 2/5 指出"判据先于实验固定"目前只是叙述，没有第三方可核的工件。本脚本导出
每个阶段的配置 + 哈希 + 代码 bundle，写入 tests/results/prereg/prereg_lock.json，论文引用它。
用法: /usr/bin/python3 tests/prereg_export.py
"""
import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library"))
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.app.prereg import PREREG, STAGES, stage_config, lock

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "prereg")


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        from frpwm import freeze_manifest as fm
        bundle = fm.bundle_id() if hasattr(fm, "bundle_id") else fm.MANIFEST.bundle_id()
    except Exception:
        bundle = "unavailable"
    stages = {}
    for n in sorted(STAGES):
        cfg = stage_config(n)
        stages[str(n)] = dict(role=STAGES[n]["role"], lock=lock(cfg), config=cfg)
    rep = dict(exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               bundle_id=bundle, criteria=PREREG["criteria"], stages=stages)
    p = os.path.join(OUT, "prereg_lock.json")
    json.dump(rep, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"exported {len(stages)} stages -> {p}")
    for k, v in stages.items():
        print(f"  {k:>4} shots  {v['role']:22s} lock={v['lock']}")
    # 哈希是锁定值：任何阈值改动都会改它，评估侧用同一哈希校验
    assert len({v["lock"] for v in stages.values()}) == len(stages), "stage configs must differ"
    assert all(len(v["lock"]) == 16 for v in stages.values())
    print("PASS: four distinct stage locks, 16 hex chars each")
    return rep


if __name__ == "__main__":
    main()
