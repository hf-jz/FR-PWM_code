"""Verify every archived record against the manifests shipped beside it.

Both manifests are checked, and a file that a manifest lists but that is not on disk is a problem,
not a silent skip: the earlier version read only the repository manifest, so seventy-one entries of
the artefact manifest pointed at files that were never shipped and nobody noticed.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artefacts")

bad = []
total = 0
for name in ("manifest.json", "manifest_repo.json"):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        bad.append((name, "manifest missing"))
        continue
    man = json.load(open(path))
    for e in man["entries"]:
        total += 1
        p = os.path.join(ART, e["path"])
        if not os.path.exists(p):
            bad.append((e["path"], "missing (%s)" % name))
            continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != e["sha256"]:
            bad.append((e["path"], "hash (%s)" % name))

print("%d entries checked across both manifests, %d problems" % (total, len(bad)))
for p, why in bad[:20]:
    print("  ", why, p)
sys.exit(1 if bad else 0)
