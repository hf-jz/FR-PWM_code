"""Verify the shipped zip matches the manifest (count, hashes, no missing folder)."""
import hashlib
import json
import zipfile

z = zipfile.ZipFile("FRPWM_Reproducibility.zip")
names = [n for n in z.namelist() if not n.endswith("/")]
print("zip entries:", len(names))
man = json.load(open("FRPWM_Reproducibility/manifest.json"))
print("manifest files:", man["files"], "bytes:", man["total_bytes"])
by = {e["path"]: e["sha256"] for e in man["entries"]}
bad = []
for n in names:
    rel = n.split("/", 1)[1] if "/" in n else n
    if rel in ("MANIFEST.sha256", "manifest.json"):
        continue
    h = hashlib.sha256(z.read(n)).hexdigest()
    if by.get(rel) != h:
        bad.append(rel)
print("hash mismatches:", bad[:5], "count", len(bad))
groups = sorted({n.split("/")[1] for n in names if n.count("/") >= 2})
print("groups:", len(groups))
for g in groups:
    print("  %-32s %d files" % (g, sum(1 for n in names if n.split("/")[1] == g)))
