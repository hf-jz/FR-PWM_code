"""Recompute the MAST replay percentages from the archived frame log (A5)."""
import csv
import statistics

rows = list(csv.DictReader(open("stage5/data/mast_frame_logs.csv")))
n = len(rows)
def truth(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")

appr = sum(truth(r["approved"]) for r in rows)
rej = sum(truth(r["rejected"]) for r in rows)
held = sum(truth(r["held"]) for r in rows)
print("frames", n, "approved %.1f%% rejected %.1f%% held %.1f%%" % (100*appr/n, 100*rej/n, 100*held/n))

by = {}
for r in rows:
    by.setdefault(r["discharge_id"], []).append(r)
dec = [sum(truth(x["approved"]) + truth(x["rejected"]) for x in v) / len(v) for v in by.values()]
app_all = [sum(truth(x["approved"]) for x in v) / len(v) for v in by.values()]
print("shots", len(by), "frames/shot", sorted({len(v) for v in by.values()}))
print("median approved/all-frames %.3f | median approved/decided %.3f" % (statistics.median(app_all), statistics.median(dec)))

checks = [r["check_status"] for r in rows]
ok = sum(1 for c in checks if c.strip().upper() in ("PASS", "OK", "TRUE"))
print("check_status distinct", sorted(set(checks))[:6], "pass rows", ok)
ood = [float(r["ood_score"]) for r in rows if r["ood_score"] not in ("", "nan")]
print("frames with OOD>0.5: %d (%.1f%%)" % (sum(o > 0.5 for o in ood), 100*sum(o > 0.5 for o in ood)/len(ood)))
