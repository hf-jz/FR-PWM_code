"""ADAS 数据获取 + PEC 计算（流水线步 7-8）。
本地 adas_full 镜像优先命中；PEC 按 (元素,价态,波长) 查 ADF15（EXCIT 优先，log 网格插值）。"""
import os, sys
from .._env import external  # noqa: E402
import numpy as np

_ADAS_FULL = external("", "ADAS_FULL", "path of the ADAS release")
_SPEC2CONC = external("", "SPEC2CONC", "path of the plasma_spec2conc package")
sys.path.insert(0, _SPEC2CONC)
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import adas_pec  # 复用已验证解析器

# 常用谱线（真实 ADAS 可查，波长 Å）
LINES = {
    "C II 903.7":  ("c", 2, 903.7),  "C III 977":   ("c", 3, 977.0),
    "C III 1175":  ("c", 3, 1175.7), "C IV 1549":   ("c", 4, 1549.1),
    "C VI 182.2":  ("c", 6, 182.2),   # dominant state at core temperatures
    # the label follows the ADAS file the transition comes from: pju#c0 is the neutral stage, so the
    # 834.4 A line reads C I, not C II. Six transitions, six consecutive ionisation stages.
    "C I 834.4":   ("c", 1, 834.4),   "C II 426.6":  ("c", 2, 426.6),
    "C III 288.4": ("c", 3, 288.4),  "C IV 222.3":  ("c", 4, 222.3),
    "C V 33.4":    ("c", 5, 33.4),
    "O II 441.7":  ("o", 2, 441.7),  "N II 505.6":  ("n", 2, 505.6),
    "N IV 765.1":  ("n", 4, 765.1),  "Ne VIII 773.7": ("ne", 8, 773.7),
    "H_alpha":     ("h", 1, 6561.9), "He II 468.6": ("he", 2, 4686.0),
}

def _adf15_dir(element, base=None):
    """本地镜像目录：pec96 系列（轻元素）/ pec93 系列，命中即跳过下载。"""
    base = base or os.path.join(_ADAS_FULL, "adf15")
    for series in (f"pec96#{element}", f"pec93#{element}", f"pec40#{element}"):
        d = os.path.join(base, series)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"本地镜像无元素 {element}（需联网下载后缓存）")

_DL = None
def _downloader():
    """复用镜像内已验证的下载器（importlib 加载，import 安全有 __main__ 守卫）。"""
    global _DL
    if _DL is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_adas", os.path.join(_ADAS_FULL, "download_adas.py"))
        _DL = importlib.util.module_from_spec(spec); spec.loader.exec_module(_DL)
    return _DL

def ensure_element(element, base=None, mod=None):
    """流水线步 7：本地镜像有该元素 ADF15 → ("local", dir) 跳过；
    无 → 从 OPEN-ADAS 下载该元素系列目录（全价态，pju/pjr 等全类）入镜像 → ("downloaded", dir)。
    复用 download_adas.py 的 fetch/file_list（detail→download URL 变换、HTML 守卫、.part 原子写）。
    base/mod 仅供测试注入。"""
    element = element.lower()
    try:
        return "local", _adf15_dir(element, base)
    except FileNotFoundError:
        pass
    m = mod or _downloader()
    out = base or os.path.join(_ADAS_FULL, "adf15")
    links = [r for r in m.file_list() if f"#{element}/" in r.replace("][", "#")]
    if not links:
        raise FileNotFoundError(f"OPEN-ADAS 列表中无元素 {element} 的 ADF15 文件")
    n = 0
    for rel in links:
        relpath = rel.replace("detail/adf15/", "", 1).replace("][", "#")
        dest = os.path.join(out, relpath)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = m.fetch(m.BASE_URL + "/" + rel.replace("detail", "download"))
        if len(data) < 1000 or b"<!DOCTYPE" in data[:500]:
            raise ValueError(f"疑似 HTML 而非 ADF 数据: {rel}")
        tmp = dest + ".part"
        open(tmp, "wb").write(data)
        os.replace(tmp, dest)
        n += 1
    return "downloaded", _adf15_dir(element, base)

def _find_file(element, wl, q):
    """在镜像目录内找含该波长的文件：优先 pju(EXCIT) 系列。"""
    d = _adf15_dir(element)
    cands = sorted(os.listdir(d))
    for pref in ("pju", "pjr", "llu", "llr", ""):
        for fn in cands:
            if pref and pref not in fn:
                continue
            p = os.path.join(d, fn)
            try:
                t = adas_pec.find_pec(p, wl, tol=0.6)
            except Exception:
                continue
            if t is not None:
                return t
    return None

class PecCache:
    """PEC 查表缓存（幂等：本地命中不重复解析）。"""
    def __init__(self):
        self._c = {}
    def pec(self, element, q, wl):
        key = (element, q, round(wl, 1))
        if key not in self._c:
            t = _find_file(element, wl, q)
            if t is None:
                raise FileNotFoundError(f"ADF15 缺 {element}+{q} @{wl}Å")
            self._c[key] = t
        return self._c[key]
    def pec_value(self, element, q, wl, ne_cm3, te_eV):
        """PEC_q(λ; ne,Te) [ph·cm³/s]（PecTable.__call__ 已返回线性值，仅标量，向量化包装）。"""
        t = self.pec(element, q, wl)
        f = np.vectorize(lambda a, b: float(t(float(a), float(b))))
        return np.asarray(f(ne_cm3, te_eV), dtype=float)

def ion_fractions(element, ne_cm3, te_eV):
    """电离平衡丰度（ADF11 scd/acd，步 8 辅助）：→ {q: frac}（复用已验证 ion_frac）。"""
    return adas_pec.ion_frac(element, ne_cm3, te_eV)

if __name__ == "__main__":
    pc = PecCache()
    v = pc.pec_value("c", 3, 977.0, 1e14, 30.0)
    assert 1e-12 < v < 1e-6, v
    fr = dict(ion_fractions("c", 1e14, 800.0))
    assert abs(sum(fr.values()) - 1.0) < 0.05 and fr[max(fr, key=fr.get)] > 0.5
    print("adas OK:", v, "C@800eV 主态 q=", max(fr, key=fr.get))
