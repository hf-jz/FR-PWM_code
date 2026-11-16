"""诊断弦坐标读取（流水线步 6）：弦视线参数 → 采样网格。"""
import json
import numpy as np

def load_chords(path):
    """JSON: [{"name","R0","Z0","dR","dZ","L"}]（起点+方向+弦长）。"""
    with open(path) as f:
        return json.load(f)

def default_chords(R0=1.7, a=0.6):
    """MVP-1 默认：中平面 5 条水平弦（可见轫致路线），覆盖 ρ≈0-0.9。"""
    return [dict(name=f"mid{i+1}", R0=R0 - a + 2 * a * i / 4, Z0=0.0,
                 dR=0.0, dZ=1.0, L=2.5) for i in range(5)]

def sample_chord(ch, n=60):
    """弦采样网格 → (R(s), Z(s), ds)。n=60 兼顾边缘壳层分辨与在线预算（200 点会使 PEC 标量查表 20ms+/步）。"""
    s = np.linspace(-ch["L"] / 2, ch["L"] / 2, n)
    d = np.hypot(ch["dR"], ch["dZ"])
    R = ch["R0"] + s * ch["dR"] / d
    Z = ch["Z0"] + s * ch["dZ"] / d
    return R, Z, abs(ch["L"]) / n * 100.0   # ds in cm

if __name__ == "__main__":
    ch = default_chords()[2]
    R, Z, ds = sample_chord(ch)
    assert R.size == 60 and ds > 0 and abs(Z.mean()) < 1e-9
    print("chords OK")
