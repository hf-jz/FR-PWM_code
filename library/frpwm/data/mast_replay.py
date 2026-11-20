"""真实 MAST shot 回放植物（tokamark-v1 → FR-PWM 流水线系统测试）。
接口与 SyntheticPlant 一致（frame(t)/accept/u_heat/c_frac/elm_flag），
标量全部来自真实信号；剖面=参数化形状锚定真实 ne_bar（TokaMark 无 TS 剖面）。
假设显式标注：Te0=600eV 常量（MAST 欧姆放电典型）、c_frac=0.03（真实杂质未知，
supervisor 的 r_y 比较对此假设仅作链路自检，不作物理结论）。"""
import os
import numpy as np
from .mast import load_shot
from ..physics.profiles import rho_grid, typical_profiles
from ..physics.gfile import generate_gfile
from ..physics.fluxmap import FluxMap
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "paths.py")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from paths import ARTEFACTS, OUT, record  # noqa: E402
from ..physics.chords import default_chords

def _interp(t_src, y_src, t, default=0.0):
    if t_src is None or len(t_src) < 2:
        return default
    return float(np.interp(t, t_src, y_src, left=y_src[0], right=y_src[-1]))

def _highpass(t, y, win=31):
    if len(y) < win:
        return np.zeros_like(y)
    k = np.ones(win) / win
    trend = np.convolve(y, k, mode="same")
    return y - trend

class MastReplayPlant:
    """按真实 shot 时间序列回放。frame(None) 顺序推进；frame(t) 按时刻取值。"""
    def __init__(self, zpath, te0=600.0, c_frac=0.03):
        self.zpath = zpath
        self.shot_id = int(os.path.basename(zpath).split(".")[0])
        sig = load_shot(zpath)
        assert "ne" in sig and "ip" in sig, f"{zpath} 缺 ne/ip 主信号"
        self._sig = sig
        t_ne = sig["ne"][0]
        self.t_grid = t_ne[(t_ne >= max(0.0, t_ne[0]))]
        self.t0, self.t1 = float(self.t_grid[0]), float(self.t_grid[-1])
        # MHD 代理：ne 高通起伏（无 dalpha/磁信号可用时的诚实代理，非物理振幅）
        t_h, y_h = sig["ne"]
        hp = np.abs(_highpass(t_h, y_h))
        self._hp = (t_h, hp / max(hp.max(), 1e-30) * 0.15)
        ne = sig["ne"]
        self._dne = (ne[0], np.gradient(ne[1], ne[0]))
        ip = sig["ip"]
        self._dip = (ip[0], np.gradient(ip[1], ip[0]))
        self.rho = rho_grid()
        # 物理链所需几何（与 SyntheticPlant 同构；MAST 尺寸 R0≈0.85m, a≈0.55m）
        self.g = generate_gfile(R0=0.85, a=0.55)
        self.fm = FluxMap(self.g)
        self.chords = default_chords(0.85, 0.55)
        self.rng = np.random.default_rng(0)
        self.u_heat = 0.0
        self.c_frac = c_frac
        self.te0 = te0
        self.elm_flag = False
        self._idx = 0
    def next_t(self):
        """顺序回放的下一时刻（到尾后回到开头循环）。"""
        t = float(self.t_grid[self._idx])
        self._idx = (self._idx + 1) % len(self.t_grid)
        return t
    def _scalar(self, key, t, default=0.0):
        s = self._sig.get(key)
        return _interp(s[0], s[1], t, default) if s else default
    def frame(self, t=None):
        t = self.next_t() if t is None else float(t)
        ne_bar = max(self._scalar("ne", t, 1e18), 5e17)
        Te0, ne0, Ti0 = typical_profiles(self.rho)
        Te = Te0 / Te0[0] * self.te0                       # 形状保真，幅度=假设 Te0
        ne = ne0 / ne0[0] * (2.0 * ne_bar)                 # 抛物线芯部≈2×线平均
        Ti = 0.6 * Te
        mhd = _interp(*self._hp, t, 0.0)
        self.elm_flag = mhd > 0.06
        return dict(t=t, Te=Te, ne=ne, Ti=Ti, rho=self.rho,
                    mhd_amp=mhd,
                    beta_N=max(self._scalar("beta_N", t, 1.0), 0.1),
                    q95=4.2,                                   # TokaMark 无 q95 → 默认（假设）
                    rad_frac=float(np.clip(self._scalar("p_rad", t, 0.0)
                                           / max(self._scalar("p_rad", self.t1, 1.0), 1.0) * 0.8, 0, 0.95)),
                    dIp_dt=_interp(*self._dip, t, 0.0) / 1e6,  # MA/s
                    d2e_dt=_interp(*self._dne, t, 0.0) / 1e19)
    def accept(self, u):
        """回放为数据驱动：记录控制输入但不改变物理（开环观测语义）。"""
        self.u_heat = float(np.clip(u, -1, 1))

class MastReplayCarousel:
    """多炮轮播（演示模式）：顺序推进当前炮，回卷时切下一炮（循环）。
    与 MastReplayPlant 同接口；几何共享（同装置），elm_flag/u_heat/c_frac 委托当前炮。"""
    def __init__(self, zpaths, **kw):
        assert zpaths, "空炮单"
        self.plants = [MastReplayPlant(zp, **kw) for zp in zpaths]
        self.i = 0
        self._prev_t = None
        self.shots_done = 0
    @property
    def cur(self):
        return self.plants[self.i]
    def __getattr__(self, k):                     # chords/fm/g/rng/rho/c_frac/t0/t1 等委托
        return getattr(self.cur, k)
    @property
    def shot_id(self):
        return self.cur.shot_id
    def next_t(self):
        t = self.cur.next_t()
        if self._prev_t is not None and t < self._prev_t - 1e-9:   # 回卷 → 切炮
            self.i = (self.i + 1) % len(self.plants)
            self.shots_done += 1
            t = self.cur.next_t()
        self._prev_t = t
        return t
    def frame(self, t=None):
        return self.cur.frame(t)
    def accept(self, u):
        self.cur.accept(u)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library"))
    from frpwm.runtime.pipeline import FrpwmMVP
    plant = MastReplayPlant("/tmp/mast_zarr/11766.zarr")
    app = FrpwmMVP(plant=plant)
    n = 40
    ts = np.linspace(plant.t0, plant.t1, n)
    out = [app.step(t=float(x)) for x in ts]
    bad = [s for s in out if not np.isfinite(s["c_est"] or np.nan)]
    assert not bad and all(0.0 <= s["Q"] <= 1.0 for s in out)
    assert all(s["decision"]["status"] in ("approved", "corrected", "rejected") for s in out)
    h = out[-1]["health"]
    assert h.get("physics_chain") == "ok", h
    lat = max(s["latency_ms"] for s in out)
    print(f"mast_replay OK: shot {plant.shot_id} {n} 帧回放, t=[{out[0]['t']:.3f},{out[-1]['t']:.3f}]s, "
          f"Z_eff^LOS={out[-1]['zeff_los']:.2f}, Q={out[-1]['Q']:.2f}, max {lat:.1f}ms, health={h}")
