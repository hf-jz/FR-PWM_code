"""Merge one artefact directory into the shipped manifests, idempotently.

    python3 tools/add_to_manifest.py A22_SD_TIF

Recomputes the digests for that directory only, replaces any stale entries for it in
artefacts/manifest.json and artefacts/manifest_repo.json, then rewrites
artefacts/MANIFEST.sha256 from manifest.json, so the three stay consistent.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artefacts")


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(sub):
    rows = []
    for root, _, files in os.walk(os.path.join(ART, sub)):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rows.append(dict(path=os.path.relpath(p, ART), bytes=os.path.getsize(p),
                             sha256=digest(p)))
    if not rows:
        raise SystemExit("no files under artefacts/%s" % sub)
    for name in ("manifest.json", "manifest_repo.json"):
        p = os.path.join(ART, name)
        man = json.load(open(p))
        keep = [e for e in man["entries"] if not e["path"].startswith(sub + "/")]
        man["entries"] = sorted(keep + rows, key=lambda e: e["path"])
        man["files"] = len(man["entries"])
        if "total_bytes" in man:
            man["total_bytes"] = sum(e["bytes"] for e in man["entries"])
        gone = [e["path"] for e in man["entries"]
                if not os.path.exists(os.path.join(ART, e["path"]))]
        if gone:
            raise SystemExit("%s lists %d files that are not on disk, first: %s"
                             % (name, len(gone), gone[:3]))
        with open(p, "w") as f:
            json.dump(man, f, indent=1)
        print("%s: %d entries (%d for %s)" % (name, man["files"], len(rows), sub))
    man = json.load(open(os.path.join(ART, "manifest.json")))
    with open(os.path.join(ART, "MANIFEST.sha256"), "w") as f:
        for e in man["entries"]:
            f.write("%s  %s\n" % (e["sha256"], e["path"]))
    print("MANIFEST.sha256 rewritten from manifest.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A22_SD_TIF")
