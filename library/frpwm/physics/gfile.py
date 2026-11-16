"""gfile 生成与解析（流水线步 2）。
解析：EFIT EQDsk ASCII（g143613 实测格式）；生成：圆截面参数化平衡 ψ(R,Z)。"""
import numpy as np, re
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from .._env import external  # noqa: E402

_EFMT = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)[EeDd][+-]\d+")

def parse_gfile(path):
    """解析 EQDsk → dict(RGRID,ZGRID,PSIRZ,RBBBS,ZBBBS,RMAXIS,ZMAXIS,PSIMAG,PSIBCY)。
    Fortran 定宽字段常无空格粘连（0.18E+01-0.21E-01...），用 E 格式正则切 token。"""
    with open(path) as f:
        first = f.readline().split()
        nw, nh = int(first[-2]), int(first[-1])
        body = f.read()
    raw = body.split()
    tok = []
    for x in raw:
        if _EFMT.fullmatch(x):
            tok.append(x)
        else:  # 粘连段按正则切
            tok.extend(_EFMT.findall(x))
    vals = np.array([float(x.replace("D", "E").replace("d", "e")) for x in tok])
    # 头 5 行标量: rdim zdim rcentr rleft zmid | rmaxis zmaxis simag sibry bcentr
    i = 0
    rdim, zdim, rcentr, rleft, zmid = vals[i:i+5]; i += 5
    rmaxis, zmaxis, simag, sibry, bcentr = vals[i:i+5]; i += 5
    i += 5 + 5 + 5  # current, simag dup, xdum, rmaxis dup, zmaxis dup, sibry dup ...
    # 标准 EQDsk 标量区为 20 个数（4 行×5）
    i = 20
    def arr(n):
        nonlocal i
        a = vals[i:i+n]; i += n; return a
    fpol, pres, ffprim, pprime = arr(nw), arr(nw), arr(nw), arr(nw)
    psirz = arr(nw * nh).reshape(nh, nw)
    qpsi = arr(nw)
    nbbbs, limitr = int(vals[i]), int(vals[i+1]); i += 2
    rbbbs, zbbbs = arr(nbbbs), arr(nbbbs)
    rgrid = np.linspace(rleft, rleft + rdim, nw)
    zgrid = np.linspace(zmid - zdim / 2, zmid + zdim / 2, nh)
    return dict(RGRID=rgrid, ZGRID=zgrid, PSIRZ=psirz, RBBBS=rbbbs, ZBBBS=zbbbs,
                RMAXIS=rmaxis, ZMAXIS=zmaxis, PSIMAG=simag, PSIBCY=sibry,
                FPOL=fpol, QPSI=qpsi)

def generate_gfile(R0=1.7, a=0.6, kappa=1.0, B0=2.0, nw=65, nh=65):
    """参数化圆/椭圆截面平衡 → 与 parse_gfile 同构 dict（合成数据/测试用）。
    ψ(R,Z) = ψ0 * ((R-R0)^2/a^2 + (Z/(a*kappa))^2)，LCFS 处 ψ=ψ_edge。"""
    rgrid = np.linspace(R0 - 1.3 * a, R0 + 1.3 * a, nw)
    zgrid = np.linspace(-1.3 * a * kappa, 1.3 * a * kappa, nh)
    R, Z = np.meshgrid(rgrid, zgrid)
    psi = ((R - R0) / a) ** 2 + (Z / (a * kappa)) ** 2   # ψ=1 即 LCFS
    th = np.linspace(0, 2 * np.pi, 200)
    return dict(RGRID=rgrid, ZGRID=zgrid, PSIRZ=psi,
                RBBBS=R0 + a * np.cos(th), ZBBBS=a * kappa * np.sin(th),
                RMAXIS=R0, ZMAXIS=0.0, PSIMAG=0.0, PSIBCY=1.0,
                FPOL=np.full(nw, R0 * B0), QPSI=np.linspace(1.2, 3.5, nw))

if __name__ == "__main__":
    g = parse_gfile(external("g143613.006000", "MAST_GILE", "directory holding the EFIT g-files"))
    assert g["PSIRZ"].shape == (129, 129) and 1.0 < g["RMAXIS"] < 2.2
    g2 = generate_gfile()
    assert g2["PSIRZ"].shape == (65, 65) and abs(g2["PSIBCY"] - 1.0) < 1e-9
    print("gfile OK: parse real + generate synthetic")
