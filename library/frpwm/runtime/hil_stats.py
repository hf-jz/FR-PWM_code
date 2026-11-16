"""FEC-06 HIL 零漏检统计上界：Nmiss=0 不得写成 P=0，必须同时报告上界。
统一输出 Nmiss / Ncase / Neff / δmiss / p_upper_miss；
聚类场景用有效独立样本数 Neff（cluster-aware）；统计方法在 Pre-registered Acceptance 预固定。"""
import math

def cp_upper(n_miss, n_case, delta=0.05):
    """Clopper-Pearson 单侧上界（独立 case）：Beta exact。
    n_miss=0 时 p_upper = 1 − δ^(1/n)（规则形式，与 exact 一致）。"""
    assert 0 <= n_miss <= n_case and 0 < delta < 1
    n = max(n_case, 1)
    if n_miss == 0:
        return 1.0 - delta ** (1.0 / n)
    # 一般情形：Beta(n_miss+1, n−n_miss) 的 1−δ 分位上界（数值二分）
    from scipy.stats import beta
    return float(beta.ppf(1.0 - delta, n_miss + 1, n - n_miss))

def n_effective(cluster_sizes):
    """聚类有效独立样本数：Neff = (Σn_i)² / Σn_i²（design-effect 校正，保守下界取整）。"""
    cs = [max(int(c), 1) for c in cluster_sizes]
    if not cs:
        return 0
    tot = sum(cs)
    return max(1, int(tot * tot / sum(c * c for c in cs)))

def hil_report(n_miss, n_case, cluster_sizes=None, delta=0.05):
    """HIL 验收统计段（FEC-06）：聚类时按 Neff 给 cluster-aware 上界。"""
    n_eff = n_effective(cluster_sizes) if cluster_sizes else n_case
    p_up = cp_upper(n_miss, n_eff, delta)
    return dict(Nmiss=n_miss, Ncase=n_case, Neff=n_eff, delta_miss=delta,
                p_upper_miss=p_up,
                method="Clopper-Pearson exact (Neff cluster-aware)"
                if cluster_sizes else "Clopper-Pearson exact")

if __name__ == "__main__":
    r = hil_report(0, 2000)
    assert abs(r["p_upper_miss"] - (1 - 0.05 ** (1 / 2000))) < 1e-12
    assert r["p_upper_miss"] < 0.002
    r2 = hil_report(0, 2000, cluster_sizes=[100] * 20)
    assert r2["Neff"] == 20 and r2["p_upper_miss"] > r["p_upper_miss"]   # 聚类 → 上界更保守
    r3 = hil_report(1, 2000)
    assert r3["p_upper_miss"] > 0
    print(f"hil_stats OK: Nmiss=0/2000 p_upper={r['p_upper_miss']:.2e}, "
          f"聚类 Neff={r2['Neff']} p_upper={r2['p_upper_miss']:.2e}")
