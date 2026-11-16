"""Verify every archived record against the manifest shipped beside it."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(HERE, "artefacts", "manifest_repo.json")))
bad = []
for e in man["entries"]:
    p = os.path.join(HERE, "artefacts", e["path"])
    if not os.path.exists(p):
        bad.append((e["path"], "missing"))
        continue
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != e["sha256"]:
        bad.append((e["path"], "hash"))
print("%d files checked, %d problems" % (len(man["entries"]), len(bad)))
for p, why in bad[:10]:
    print("  ", why, p)
sys.exit(1 if bad else 0)
