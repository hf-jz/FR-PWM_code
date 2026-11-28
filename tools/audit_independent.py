"""Independent re-computation of the identifiability numbers, using nothing from this package.

The point is to be able to disagree with `analysis.sdtif`.  This file imports numpy, json, os and
hashlib and nothing else: no `Problem`, no `localize`, no `analyze_target`.  It reads the archived
operators out of A22, rebuilds the whitened matrix, redoes the decomposition itself, and compares
what it gets with what the record says.  Anything it cannot reproduce is reported as a failure with
the paper location that quotes it.

Checks
  1  manifest     every recorded file hashes to the digest the manifest claims
  2  spectra      stored singular values reproduce from the stored operator
  3  ranks        structural and practical ranks reproduce, and their thresholds are ordered
  4  occupancy    rank <= occupied for every node of the weighted operator
  5  projectors   idempotent, symmetric, and they annihilate the operator
  6  targets      Gamma(N_tot), Gamma(N_obs) reproduce from the stored operator
  7  robustness   the Davis-Kahan bound exceeds the measured drift at every perturbation size

Usage
  python3 tools/audit_independent.py            # summary on stdout, record in A24
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artefacts")
A22 = os.path.join(ART, "A22_SD_TIF")
A24 = os.path.join(ART, "A24_Independent_Audit")

FAILS = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=detail))
    if not ok:
        FAILS.append((name, detail))
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------- 1 manifest
def audit_manifest():
    man = json.load(open(os.path.join(ART, "manifest.json")))
    bad, missing = [], []
    for e in man["entries"]:
        p = os.path.join(ART, e["path"])
        if not os.path.exists(p):
            missing.append(e["path"])
        elif sha256(p) != e["sha256"]:
            bad.append(e["path"])
    check("manifest.entries", len(man["entries"]) == man["files"],
          "%d listed, %d claimed" % (len(man["entries"]), man["files"]))
    check("manifest.hash_matches", not bad, "mismatched: %s" % bad[:5])
    check("manifest.files_present", not missing, "missing: %s" % missing[:5])
    return dict(files=len(man["entries"]), mismatched=bad, missing=missing)


# ------------------------------------------------------------------- 2-6 one carbon node
def whitened(row):
    """Rebuild J~ = L^{-1} A S from the stored operator, scale and noise. Nothing else is used."""
    A = np.asarray(row["matrix"], float)
    S = np.diag(np.asarray(row["scales"], float))
    sig = np.asarray(row["noise_floor"], float).ravel()
    return np.diag(1.0 / sig) @ A @ S


def audit_node(row, targets):
    J = whitened(row)
    n = J.shape[1]
    # svd returns min(m,n) numbers; the record keeps all n, with the missing ones as zeros
    s = np.linalg.svd(J, compute_uv=False)
    s = np.r_[s, np.zeros(max(0, n - s.size))]
    s1 = s[0]
    tau_num = max(J.shape) * np.finfo(float).eps * s1
    tau_prac = 1.0 / row["eps_req"]
    r_num = int(np.sum(s > tau_num))
    r_prac = int(np.sum(s > tau_prac))
    U, S, Vt = np.linalg.svd(J)
    V0 = Vt.T[:, r_num:]
    P0 = V0 @ V0.T if V0.size else np.zeros((n, n))

    tag = "%.0f eV, %.0e" % (row["te"], row["ne"])
    check("spectra[%s]" % tag,
          np.allclose(s, row["singular_values"], rtol=1e-8, atol=1e-14),
          "max rel diff %.2e" % (np.max(np.abs(s - np.asarray(row["singular_values"])))
                                 / max(s1, 1e-300)))
    check("rank_num[%s]" % tag, r_num == row["rank_num"], "%d vs stored %d" % (r_num, row["rank_num"]))
    check("rank_prac[%s]" % tag, r_prac == row["rank_prac"], "%d vs stored %d" % (r_prac, row["rank_prac"]))
    check("thresholds_ordered[%s]" % tag, tau_prac >= tau_num,
          "tau_prac %.2e, tau_num %.2e" % (tau_prac, tau_num))
    check("occupancy[%s]" % tag, row["rank_num"] <= row["occupied"],
          "rank %d, occupied %d" % (row["rank_num"], row["occupied"]))
    check("nullity[%s]" % tag, n - r_num == len(row["null_basis"]),
          "%d vs %d vectors" % (n - r_num, len(row["null_basis"])))
    check("projector_idempotent[%s]" % tag, np.allclose(P0 @ P0, P0, atol=1e-10))
    check("projector_symmetric[%s]" % tag, np.allclose(P0, P0.T, atol=1e-12))
    check("projector_annihilates[%s]" % tag,
          np.linalg.norm(J @ P0) <= max(tau_num, 1e-12),
          "||J P0|| = %.3e, tau_num %.3e" % (np.linalg.norm(J @ P0), tau_num))

    # the raw decomposition is audited too, because the table prints its rank: keeping a column
    # twenty decades below the noise is the defect the table exists to expose, and it has to be
    # reproducible rather than asserted
    A_raw = np.asarray(row["matrix_raw"], float)
    sig = np.asarray(row["noise_floor"], float).ravel()
    # structural count: a column with any nonzero entry is a column, no threshold involved
    r_raw = int(min(A_raw.shape[0], np.sum(np.any(A_raw != 0.0, axis=0))))
    check("rank_raw[%s]" % tag, r_raw == row["rank_raw"], "%d vs stored %d" % (r_raw, row["rank_raw"]))
    check("raw_exceeds_occupied_flag[%s]" % tag,
          int(r_raw > row["occupied"]) == row["rank_raw_exceeds_occupied"],
          "flag disagrees with the numbers")

    out = {}
    for name, c in targets(row).items():
        g = float(np.linalg.norm(P0 @ c) / np.linalg.norm(c))
        stored = row["targets"][name]["gamma"]
        check("gamma_%s[%s]" % (name, tag), abs(g - stored) < 1e-8,
              "%.10f vs stored %.10f" % (g, stored))
        check("gamma_in_unit[%s,%s]" % (tag, name), -1e-12 <= g <= 1 + 1e-12, "%.6f" % g)
        out[name] = g
    return dict(node=tag, rank=r_num, rank_prac=r_prac, nullity=n - r_num, gamma=out)


def carbon_targets(row):
    """Target gradients in the dimensionless coordinate, rebuilt from the definition.

    theta = S z and S = diag(scales), so a target with gradient g_theta has c_z = S g_theta.
    N_tot sums all seven stages; N_obs sums the six stages a line observes.
    """
    S = np.asarray(row["scales"], float)
    return dict(N_tot=S * np.ones(7), N_obs=S * np.array([1.0] * 6 + [0.0]))


# ------------------------------------------------------------------------------- 7 E6 bound
def audit_robustness(bench):
    e6 = bench["E6 Weyl / Davis-Kahan robustness"]
    gap = e6["gap"]
    worst = None
    for pt in e6["sweep"]:
        bound = pt["delta"] / gap
        check("dk_recompute[delta=%.0e]" % pt["delta"], abs(bound - pt["bound"]) < 1e-12,
              "%.6e vs %.6e" % (bound, pt["bound"]))
        check("dk_holds[delta=%.0e]" % pt["delta"], pt["drift"] <= bound,
              "drift %.4e, bound %.4e" % (pt["drift"], bound))
        margin = bound / max(pt["drift"], 1e-300)
        if worst is None or margin < worst[1]:
            worst = (pt["delta"], margin)
    check("dk_rank_preserved", all(pt["rank_preserved"] for pt in e6["sweep"]),
          "the bound is only stated where the dimension is unchanged")
    return dict(gap=gap, tightest_at=worst[0], tightest_margin=worst[1],
                bound_holds_everywhere=e6["bound_holds_everywhere"])


def main():
    os.makedirs(A24, exist_ok=True)
    man = audit_manifest()
    bench = json.load(open(os.path.join(A22, "benchmarks.json")))
    cases = json.load(open(os.path.join(A22, "cases.json")))

    nodes = [audit_node(r, carbon_targets) for r in cases["case_A_weighted"]]
    check("nodes_audited", len(nodes) == 20, "%d nodes" % len(nodes))
    rob = audit_robustness(bench)

    report = dict(nodes=nodes, manifest=man, robustness=rob,
                  checks=CHECKS, n_checks=len(CHECKS), n_failed=len(FAILS),
                  failed=[dict(check=c["check"], detail=c["detail"])
                          for c in CHECKS if not c["ok"]])
    json.dump(report, open(os.path.join(A24, "report.json"), "w"), indent=1)

    print("checks run: %d   failed: %d" % (len(CHECKS), len(FAILS)))
    for name, detail in FAILS:
        print("  FAIL %-40s %s" % (name, detail))
    print("weighted nodes: %d, max rank %d, occupancy consistent: %s"
          % (len(nodes), max(n["rank"] for n in nodes),
             all(c["ok"] for c in CHECKS if c["check"].startswith("occupancy"))))
    print("davis-kahan: gap %.0f, tightest margin %.2f at delta %.1e"
          % (rob["gap"], rob["tightest_margin"], rob["tightest_at"]))
    print("-> %s" % A24)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
