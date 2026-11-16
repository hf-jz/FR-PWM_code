"""W（钨）EUV 谱线表提取：扫描本地 ADF15 pec40#w 镜像 (W27+–W45+, ca=excitation)，
取指定 EUV 波段内各价态 top-K 强线 → W_LINES，合并入 adas.LINES 供 PecCache/谱合成使用。
用法: from frpwm.physics.wlines import w_lines; LINES.update(w_lines())"""
import os, re, json
from frpwm.physics import adas as _adas
import adas_pec  # noqa: E402  (路径由 adas 模块注入)

_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "results", "wlines_cache.json")


def scan_w_lines(band=(40.0, 65.0), stages=range(27, 46), top_k=4, ne=1e13, te=2000.0):
    """返回 {label: ("w", q, wl)}；按参考点 (ne,Te) 峰值 PEC 排序取 top_k/价态。"""
    d = _adas._adf15_dir("w")
    hdr = re.compile(r"^\s*([\d.]+)\s+\d+\s+\d+\s*/.*type=excit")
    out = {}
    for q in stages:
        f = os.path.join(d, f"pec40#w_ca#w{q}.dat")
        if not os.path.exists(f):
            continue
        cands = []
        for ln in open(f):
            m = hdr.match(ln)
            if m:
                wl = float(m.group(1))
                if band[0] <= wl <= band[1]:
                    cands.append(round(wl, 3))
        rated = []
        for wl in sorted(set(cands)):
            try:
                t = adas_pec.find_pec(f, wl, tol=0.05)
                if t is not None:
                    rated.append((float(t(ne, te)), wl))
            except Exception:
                continue
        rated.sort(reverse=True)
        for rank, (pk, wl) in enumerate(rated[:top_k]):
            out[f"W {q}+ {wl:.2f}"] = ("w", q, wl)
    return out


def w_lines(band=(40.0, 65.0), stages=range(27, 46), top_k=4, use_cache=True):
    """幂等：缓存命中直接返回；否则扫描并写缓存。"""
    key = f"{band[0]:.0f}-{band[1]:.0f}_{min(stages)}-{max(stages)}_k{top_k}"
    if use_cache and os.path.exists(_CACHE):
        c = json.load(open(_CACHE))
        if key in c:
            return {k: (v[0], v[1], v[2]) for k, v in c[key].items()}
    tab = scan_w_lines(band, stages, top_k)
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    c = json.load(open(_CACHE)) if os.path.exists(_CACHE) else {}
    c[key] = {k: list(v) for k, v in tab.items()}
    json.dump(c, open(_CACHE, "w"), indent=1)
    return tab


if __name__ == "__main__":
    import time
    t0 = time.time()
    tab = w_lines(use_cache=False)
    print(f"W EUV 线表: {len(tab)} 条, {time.time()-t0:.1f}s")
    for k, v in list(tab.items())[:12]:
        print(" ", k, v)
