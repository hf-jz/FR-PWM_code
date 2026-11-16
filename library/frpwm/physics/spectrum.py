"""光谱合成（流水线步 5/9/10）：线辐射强度 + 弦积分 + 谱线重建 + 可见轫致。"""
import numpy as np
from .adas import LINES, PecCache

HC = 6.626e-34 * 3.0e8 * 1e10 / 1.602e-19   # hc in eV·Å

class SpectrumSynth:
    def __init__(self, pec_cache=None):
        self.pc = pec_cache or PecCache()

    def line_emissivity(self, element, q, wl, ne_cm3, te_eV, nz_cm3):
        """步 9：ε_λ = n_e·n_z^q·PEC_q  [ph/(cm³·s)]。"""
        return ne_cm3 * nz_cm3 * self.pc.pec_value(element, q, wl, ne_cm3, te_eV)

    def los_integrate(self, eps_s, ds_cm):
        """步 5：弦积分 I = ∫ε ds [ph/(cm²·s·sr-ish)]（几何因子并入标定）。"""
        return float(np.trapz(eps_s, dx=ds_cm))

    def bremsstrahlung(self, ne_cm3, te_eV, zeff, wl_ang=5320.0):
        """可见轫致发射率 ∝ n_e²·Z_eff·g_ff·exp(-hc/λTe)/√Te（任意单位）。"""
        gff = 1.2
        return ne_cm3**2 * zeff * gff * np.exp(-HC / (wl_ang * np.maximum(te_eV, 1.0))) / np.sqrt(te_eV)

    def reconstruct(self, wl_grid, lines_intensity, fwhm_ang=2.0, cont_level=0.0):
        """步 10：谱线重建——高斯线型叠加 + 连续谱本底 + 仪器响应（高斯卷积近似并入线宽）。"""
        spec = np.full_like(wl_grid, cont_level, dtype=float)
        for wl0, inten in lines_intensity.items():
            sig = fwhm_ang / 2.355
            spec += inten / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl_grid - wl0) / sig) ** 2)
        return spec

    def synth_chord_spectrum(self, chord_lines, ne_s, te_s, nz_s, ds_cm):
        """一条弦上若干谱线的弦积分强度 → {name: I}。ne_s/te_s/nz_s 为弦采样数组。"""
        out = {}
        for name in chord_lines:
            el, q, wl = LINES[name]
            eps = self.line_emissivity(el, q, wl, ne_s, te_s, nz_s)
            out[name] = self.los_integrate(eps, ds_cm)
        return out

if __name__ == "__main__":
    ss = SpectrumSynth()
    ne = np.full(100, 1e14); te = np.full(100, 30.0); nz = np.full(100, 1e10)
    d = ss.synth_chord_spectrum(["C III 977", "C IV 1549"], ne, te, nz, 1.0)
    assert d["C III 977"] > 0 and d["C IV 1549"] > 0
    g = ss.reconstruct(np.linspace(970, 990, 400), {977.0: 1e12})
    assert abs(g.max() - 1e12 / (2.355 * (2.0 / 2.355) * np.sqrt(2 * np.pi)) * 2.355) / g.max() < 0.01
    print("spectrum OK:", d)
