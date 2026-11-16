"""STRAHL 批量训练库生成器：LHS 采样 (Dmax,decay,v) → 并行 run_forward_dv → 监督样本。
样本 = {params: (Dmax,decay,v), r, n_tot(末时刻), D(ρ)/V(ρ) 解析标签, frame(可选全帧谱)}。
背景 Te/ne 剖面从 strahl base pp00020.0 解析（真实 HL-2A 背景，所有工况共享）。
用法: /usr/bin/python3 -m frpwm.data.strahl_library c 64 4  (元素 n_cases n_workers)"""
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from frpwm.physics.transport import run_forward_dv

_PP = external("nete/pp00020.0", "STRAHL_ROOT", "path of the STRAHL source tree")
_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "results", "library")


def parse_pp(path=_PP):
    """strahl pp 文件 → (rho, ne_cm3, te_eV)。布局: x-grid(# pts)→DATA 块 ×Ne/Te。"""
    lines = open(path).read().split("\n")
    def floats_after(idx):
        vals = []
        i = idx + 1
        while i < len(lines) and (lines[i].strip() == "" or not lines[i].lstrip().startswith("cv")):
            vals += [float(x) for x in lines[i].split()]
            i += 1
        return vals
    anchors = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if "x-grid for Ne" in ln:
            anchors["rho_ne"] = floats_after(i)
        elif "DATA (cm**-3)" in ln:
            anchors["ne"] = floats_after(i)
        elif "x-grid for Te" in ln:
            anchors["rho_te"] = floats_after(i)
        elif s == "cv    DATA" and "te" not in anchors:
            anchors["te"] = floats_after(i)
    rho = np.array(anchors["rho_ne"]); ne_raw = np.array(anchors["ne"])
    rho_te = np.array(anchors["rho_te"]); te_raw = np.array(anchors["te"])
    # 'interpa' 惯例: DATA[0]=中心值(head), DATA[1:]=逐 ρ 形状因子(shape[0]=1) → 绝对值 = head×shape
    ne = ne_raw[0] * ne_raw[1:1 + len(rho)]
    te = te_raw[0] * te_raw[1:1 + len(rho_te)]
    return rho, ne, rho_te, te


def lhs_params(n, seed=7, dmax=(0.5, 8.0), decay=(2.0, 8.0), v=(0.0, 30.0)):
    """拉丁超立方 3 参数。"""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 3))
    for j, (lo, hi) in enumerate((dmax, decay, v)):
        perm = rng.permutation(n)
        P[:, j] = (perm + rng.random(n)) / n * (hi - lo) + lo
    return P


def _one(args):
    idx, dmax, decay, v, element, aw, timeout = args[:7]
    n_time = args[7] if len(args) > 7 else 0
    try:
        out = run_forward_dv(dmax, decay, v, element=element, atomic_weight=aw, timeout=timeout,
                             n_time=n_time)
    except Exception:
        out = None                      # 超时/崩溃=失败工况, 不炸整库（旧版 TimeoutExpired 传播曾团灭 W-12 首跑）
    if out is None:
        return dict(idx=idx, ok=False, params=[dmax, decay, v])
    d = dict(idx=idx, ok=True, params=[dmax, decay, v],
             t=out["t"], r=out["r"].tolist(), n_tot=out["n_tot"].tolist())
    if "n_tot_t" in out:
        d.update(ts=out["ts"].tolist(), n_tot_t=out["n_tot_t"].tolist())
    return d


def gen_library(n_cases=64, element="'C_'", aw=12.0, seed=7, n_workers=4, timeout=300,
                tag=None, frames=False, n_time=0):
    """生成并落盘 json（增量写盘）。n_time>0 时保存时间分辨剖面（T=n_time）。"""
    t0 = time.time()
    P = lhs_params(n_cases, seed)
    args = [(i, float(p[0]), float(p[1]), float(p[2]), element, aw, timeout, n_time)
            for i, p in enumerate(P)]
    res = []
    os.makedirs(_OUT, exist_ok=True)
    tag = tag or f"{element.strip(chr(39)).lower()}{n_cases}_s{seed}"
    path = os.path.join(_OUT, f"library_{tag}.json")

    def dump(final=False):
        json.dump(dict(element=element, aw=aw, seed=seed, n_cases=n_cases,
                       n_ok=sum(1 for r in res if r["ok"]), cases=res,
                       partial=not final), open(path, "w"))

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_one, args):
            res.append(r)
            if len(res) % 8 == 0:
                print(f"  [{len(res)}/{n_cases}] {time.time()-t0:.0f}s", flush=True)
            if len(res) % 16 == 0:
                dump()                          # 增量落盘: 进程被杀不丢已算工况
    dump(final=True)
    print(f"library {tag}: {sum(1 for r in res if r['ok'])}/{n_cases} ok, {time.time()-t0:.0f}s → {path}")
    return path, sum(1 for r in res if r["ok"])


if __name__ == "__main__":
    el = sys.argv[1] if len(sys.argv) > 1 else "c"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    nt = int(sys.argv[4]) if len(sys.argv) > 4 else 0          # n_time: 时间分辨点数
    tg = sys.argv[5] if len(sys.argv) > 5 else None
    element, aw = ("'W_'", 184.0) if el.lower().startswith("w") else ("'C_'", 12.0)
    gen_library(n, element, aw, n_workers=nw, timeout=1500 if aw > 100 else 300,
                n_time=nt, tag=tg)
