"""strahl-cpp 输运正演接入（P0-1，报告 17 模块 6 / 三级求解器 L3 离线锚）。
复用 strahl-cpp/python/fusion_forward 已验证封装；在线链不直接调（秒级），
作为 L2 离线锚与解析链交叉校验（电荷态分布/浓度一致性）。"""
import os, sys
from .._env import external  # noqa: E402
import numpy as np

_REPO = external("", "STRAHL_CPP", "path of the strahl-cpp build")
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, os.path.join(_REPO, "build"))
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from fusion_forward import run_forward, DEFAULT_BASE   # noqa: E402

def run_transport(element="C", shot=20, base=DEFAULT_BASE, **kw):
    """strahl-cpp 正演 → dict(r, n_zq[t,z,r], charge_state, n_tot, p_rad, time)。"""
    ds = run_forward(element=element, shot=shot, base=base, **kw)
    return dict(r=ds["r"].values, n_zq=ds["n"].values,
                q=ds["charge_state"].values if "charge_state" in ds else np.arange(ds["n"].shape[1]),
                n_tot=ds["n_tot"].values, p_rad=ds["p_rad"].values,
                time=ds["time"].values if "time" in ds else None)

def cross_validate(fr_c_table, rho, te_tab, te_prof, element="C"):
    """解析链电荷态分布 vs strahl 高保真：主态一致性与形状相关。
    fr_c_table: {q: frac(Te)} 解析表；strahl 末时刻各态剖面 → 在相同 Te 网格上比较主态分数。"""
    out = run_transport(element)
    n_last = out["n_zq"][-1]                       # (z, r) 末时刻
    frac = n_last / np.maximum(n_last.sum(0, keepdims=True), 1e-30)
    q_dom_strahl = np.argmax(frac.mean(1))
    fr_c = np.array([np.interp(te_prof, te_tab, fr_c_table[q]) for q in sorted(fr_c_table)])
    q_dom_analytic = int(np.argmax(fr_c.mean(1)))
    return dict(q_dom_strahl=int(q_dom_strahl), q_dom_analytic=q_dom_analytic,
                frac_strahl=frac.mean(1), n_tot_last=out["n_tot"][-1])

# ---- 参数级 D/V 正演（strahl-cpp 反演关键工具：编辑 Example param 的 D 剖面/外区对流后跑二进制） ----
_CPP_BIN = os.path.join(_REPO, "build", "strahl_cpp")

def run_forward_dv(dmax, decay, v_outer, base=DEFAULT_BASE, workdir=None, timeout=300,
                   element="'C_'", atomic_weight=12.0, n_time=0):
    """编辑 param（D 剖面 Dmax/decay + 外区对流 v_outer + 可选换元素）→ strahl_cpp → out_profiles.csv。
    → dict(r, n_tot, p_rad, t)（末时刻）；无效工况返回 None。死参数说明见 ai4fusion-inversion 探针结论：
    本几何下 Dmax/decay_in/v 强有效（±50-85%），D_in/D_out/r1/decay_out 为死参数。
    element="'W_'"/atomic_weight=184.0 → 钨 75 价态通道（W_.atomdat+newdat W 速率齐备）。"""
    import subprocess, shutil, tempfile
    wd = workdir or tempfile.mkdtemp(prefix=f"sdv_{dmax:.0f}_{decay:.0f}_{v_outer:.0f}_")
    os.makedirs(wd, exist_ok=True)
    for d in ("nete", "atomdat", "result", "pec_files", "source"):
        link = os.path.join(wd, d)
        if not os.path.exists(link):
            os.symlink(os.path.join(base, d), link)
    txt = open(os.path.join(base, "param_files", "Example")).read()
    out = []
    for ln in txt.splitlines():
        sp = ln.split()
        if "'C_'" in ln and "12." in ln and len(sp) == 3:              # 元素行
            ln = ln.replace("'C_'", element).replace("12.", f"{atomic_weight:.1f}", 1)
        elif "15." in ln and len(sp) == 6 and sp[3] == "15.":       # D 剖面 funct 行
            sp[0] = "1.0"; sp[1] = "1.0"; sp[2] = f"{dmax:.6g}"
            sp[3] = f"{decay:.6g}"; sp[4] = "3.0"
            ln = "         " + "\t".join(sp)
        elif ln.strip().replace(" ", "").replace("\t", "") == "0.0.":   # 外区对流参数行
            ln = "      0.     " + f"{v_outer:.6g}"
        out.append(ln)
    pfile = os.path.join(wd, "Example")
    open(pfile, "w").write("\n".join(out))
    r = subprocess.run([_CPP_BIN, pfile, base], cwd=wd, capture_output=True, timeout=timeout)
    csv = os.path.join(wd, "out_profiles.csv")
    if r.returncode != 0 or not os.path.exists(csv):
        return None
    blocks, cur = [], None
    for line in open(csv):
        if line.startswith("# t"):
            if cur:
                blocks.append(cur)
            cur = {"t": float(line.split("=")[1].strip().split()[0]), "rows": []}
        elif line.strip() and not line.startswith("#") and cur is not None:
            cur["rows"].append([float(x) for x in line.split(",")])
    if cur:
        blocks.append(cur)
    last = blocks[-1]["rows"] if blocks else None
    if not last:
        return None
    arr = np.array(last)
    out = dict(r=arr[:, 0], n_tot=arr[:, -1], t=blocks[-1]["t"], n_blocks=len(blocks))
    if n_time and len(blocks) > 1:
        idx = np.unique(np.linspace(1, len(blocks) - 1, min(n_time, len(blocks) - 1)).astype(int))
        ts = np.array([blocks[i]["t"] for i in idx])
        n_t = np.array([np.array(blocks[i]["rows"])[:, -1] for i in idx])
        out.update(ts=ts, n_tot_t=n_t)          # 时间分辨剖面 (T, R)——稳态简并解锁数据
    return out

if __name__ == "__main__":
    out = run_transport("C")
    assert out["n_zq"].ndim == 3 and (out["n_zq"] >= 0).all()
    assert out["n_tot"][-1].max() > 1e10
    fracs = out["n_zq"][-1] / np.maximum(out["n_zq"][-1].sum(0, keepdims=True), 1e-30)
    q_dom = int(np.argmax(fracs.mean(1)))
    print(f"transport OK: 主态 q={q_dom}, n_tot_max={out['n_tot'][-1].max():.2e}, "
          f"p_rad_max={out['p_rad'].max():.2e}")
