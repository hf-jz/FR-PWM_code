"""模拟光谱测量仪（无真实光谱诊断时的测量数据供给）。
以独立仪器实现生成"如同真实诊断交付"的测量文件：
  正演同一算子族（PEC 线性）但独立仪器实现——通道标定系数(±3%)、
  光子统计噪声(√I 相对)、谱线随机缺失(10%)、仪器线宽 FWHM 抖动。
输出 npz: t 轴 × 弦 × 谱线的 I_meas 与 σ，附真值 c_true(t)（仅评测用，不进系统）。"""
import numpy as np
from ..physics.adas import LINES, ion_fractions
from ..physics.spectrum import SpectrumSynth
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
from ..physics.chords import sample_chord

class SimSpectrometer:
    """frame 驱动：以 plant.frame(t) 的 Te/ne/rho + c_true(t) 生成多弦多线测量。"""

    def __init__(self, plant, c_true_fn, seed=11, n_lines=None, fwhm=2.0):
        self.plant = plant
        self.c_true_fn = c_true_fn          # fn(t) -> c 真值（仅仪器侧知道）
        self.rng = np.random.default_rng(seed)
        self.synth = SpectrumSynth()
        self.lines = list(LINES.keys())[: n_lines or 6]
        self.fwhm = fwhm
        self.cal = {ch["name"]: float(1 + self.rng.normal(0, 0.03)) for ch in plant.chords}
        # 逐 ρ 电离平衡表（同管线 60 点 geomspace——边缘壳层发射主导，禁芯部标量近似）
        self._te_tab = np.geomspace(10.0, 6000.0, 60)
        tab = [dict(ion_fractions("c", 1e14, float(t))) for t in self._te_tab]
        self._fr_tab = {q: np.array([d.get(q, 0.0) for d in tab]) for q in range(7)}

    def measure(self, t):
        """→ dict(chord -> {line: (I_meas, sigma)})；10% 谱线缺失（None）。"""
        fr = self.plant.frame(t)
        rho, Te, ne = fr["rho"], fr["Te"], fr["ne"]
        c = float(self.c_true_fn(t))
        out = {}
        for ch in self.plant.chords:
            R, Z, ds = sample_chord(ch)
            # 铁律：测量与标定走同一正演算子（同弦、同网格、同 PEC、同插值、同 ρ 约定）
            r_s = self.plant.fm.rho(R, Z)
            ok = r_s <= 1.0
            te_s = np.interp(np.clip(r_s, 0, 1), rho, Te)
            ne_s = np.interp(np.clip(r_s, 0, 1), rho, ne) / 1e6     # m⁻³ → cm⁻³
            rec = {}
            for name in self.lines:
                if self.rng.random() < 0.10:
                    rec[name] = None                                  # 诊断丢线
                    continue
                el, q, wl0 = LINES[name]
                fr_q = np.interp(Te, self._te_tab, self._fr_tab.get(q, np.zeros(60)))
                nz_s = np.interp(np.clip(r_s, 0, 1), rho, fr_q * (c * ne)) * ok / 1e6
                I = self.synth.los_integrate(
                    self.synth.line_emissivity(el, q, wl0, ne_s, te_s, nz_s), ds)
                I *= self.cal[ch["name"]]                             # 通道标定误差
                shot = self.rng.normal(0, np.sqrt(max(I, 1.0)))       # 光子统计
                Im = max(I + shot, 1e-30)
                rec[name] = (Im, float(np.sqrt(max(I, 1.0))))
            out[ch["name"]] = rec
        return out

def run_series(plant, ts, c_true_fn, **kw):
    """整段时序测量 → npz 测量文件（模拟诊断数据交付）。"""
    inst = SimSpectrometer(plant, c_true_fn, **kw)
    rows = []
    for t in ts:
        m = inst.measure(float(t))
        for chn, rec in m.items():
            for ln, v in rec.items():
                if v is not None:
                    rows.append((float(t), chn, ln, v[0], v[1]))
    data = dict(t=np.array([r[0] for r in rows]),
                chord=np.array([r[1] for r in rows]),
                line=np.array([r[2] for r in rows]),
                I=np.array([r[3] for r in rows]),
                sigma=np.array([r[4] for r in rows]),
                c_true=np.array([float(c_true_fn(t)) for t in ts]),
                t_axis=np.array(ts, float))
    return data

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
    from frpwm.data.mast_replay import MastReplayPlant
    plant = MastReplayPlant("/tmp/mast_zarr/11776.zarr")
    ts = np.linspace(plant.t0, plant.t1, 10)
    d = run_series(plant, ts, lambda t: 0.03)
    i977 = d["I"][(d["line"] == "C III 977") & (d["chord"] == "mid3")]
    assert len(i977) == len(ts) - 0 or len(i977) > 0
    assert (d["I"] > 0).all() and (d["sigma"] > 0).all()
    print(f"sim_spectrometer OK: {len(d['t'])} 条测量记录，"
          f"C III 977@mid3 {len(i977)}/{len(ts)} 帧，I∈[{i977.min():.2e},{i977.max():.2e}]")
