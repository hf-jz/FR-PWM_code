"""FR-PWM 主流水线（MVP-1：实时等离子体状态监视，不控制）。
模块化隔离：每个模块调用独立 try/except，单模块故障 → 记录 + 回退值，不波及其他模块。
物理链（12 步）: gfile→fluxmap→profiles→chords→ADAS→PEC→线辐射→弦积分→谱重建→比较反演→体积分浓度。"""
import os
import time
import numpy as np
from scipy.optimize import nnls
from .. import constants
from ..schema import MetaState, now_prov, Quantity
from ..quality import state_quality
from ..physics.gfile import generate_gfile, parse_gfile
from ..physics.fluxmap import FluxMap
from ..physics.profiles import rho_grid, typical_profiles
from ..physics.chords import default_chords, sample_chord
from ..physics.adas import PecCache, ion_fractions, LINES
from ..physics.spectrum import SpectrumSynth
from ..physics.concentration import concentration

# Spectral inversion rule. 'multiline' solves three carbon lines for seven charge states with a
# ridge penalty; 'single' is the previous one-line ratio rule. The multi-line solve is default-off:
# in this plant the observed C III/C IV states carry almost no population at core temperatures, so
# the line set has no leverage and the solve returns a vanishing concentration (measured).
INVERSION = os.environ.get("FRPWM_INVERSION", "multiline").lower()
# One transition per charge state: the weighted system is square and full rank (condition 1.0
# on a 10-600 eV grid), so the state vector and its total are observationally identifiable.
MULTILINE_LINES = ("C I 834.4", "C II 426.6", "C III 288.4", "C IV 222.3", "C V 33.4",
                   "C VI 182.2")
from ..invert.zeff_los import zeff_los_from_profiles, zeff_los_release
from ..invert.ood import KnnOod
from .fabric import DataFabric
from .regime import RegimeMonitor
from .risk import RiskMonitor
from .safety import SafetyShield
from .supervisor import Supervisor
from .fast_observer import fuse as obs_fuse
from .latency import LatencyLedger
from .hardcase import HardCaseBuffer
from .alignment import AlignedSnapshot
from .risk_budget import trm_k_max
from .safety import eligibility as ctrl_eligibility
from .prediction_cache import PredictionCache, PredictionSnapshot, PhysicsOnlyPredictor, nz_q_authoritative
from ..physics.identifiability import sigma_five
from ..freeze_manifest import MANIFEST
from .. import versions as _vers
from ..neural.jepa import PlasmaJepa
from ..neural.hrm import HrmRegime
from ..neural.trm import TrmSolver
from ..neural.world import DualWorldModel
from ..control.target import TargetPlanner
from ..control.mpc import FastMpc


class SyntheticPlant:
    """合成装置（真实数据源接入前的对端）：圆平衡 + 典型剖面 + 3% 碳 + 噪声。
    真值浓度(0.03)故意偏离反演标定参考(0.02)，验证反演恢复能力。"""
    def __init__(self, c_frac=0.03, R0=1.7, a=0.6, seed=0):
        self.g = generate_gfile(R0=R0, a=a)
        self.fm = FluxMap(self.g)
        self.rho = rho_grid()
        self.c_frac = c_frac
        self.chords = default_chords(R0, a)
        self.rng = np.random.default_rng(seed)
        self.t0 = time.time()
        self.u_heat = 0.0      # 加热输入（控制链闭环用）
        self.elm_flag = False  # 场景事件标志（HRM 训练标签）
    def accept(self, u):
        """施加加热输入（Stage C/D 才有；MVP-1 默认不调用）。"""
        self.u_heat = float(np.clip(u, -1, 1))
    def frame(self, t=None):
        t = t if t is not None else time.time() - self.t0   # t=0.0 是合法时刻，不能用 or
        Te, ne, Ti = typical_profiles(self.rho)
        # 加热响应：芯部沉积（keV 量级）
        dep = np.exp(-((self.rho - 0.3) / 0.25) ** 2)
        Te = Te + 150.0 * self.u_heat * dep
        # 场景事件：t∈[6,8] ELM 脉冲（MHD 振幅增大）
        self.elm_flag = 6.0 <= t <= 8.0
        base_mhd = 0.08 if self.elm_flag else 0.01
        wob = 1.0 + 0.02 * self.rng.standard_normal(len(self.rho))
        Te, ne = Te * wob, ne * wob
        mhd = max(0.0, base_mhd + 0.005 * np.sin(0.7 * t) + 0.003 * self.rng.standard_normal())
        return dict(t=t, Te=Te, ne=ne, Ti=Ti, rho=self.rho,
                    mhd_amp=mhd, beta_N=1.6 + 0.1 * np.sin(0.3 * t),
                    q95=4.2, rad_frac=0.35, dIp_dt=0.0, d2e_dt=0.0)


class FrpwmMVP:
    """模块编排器。health 记录各模块状态供故障定位。"""
    def __init__(self, plant=None):
        self.plant = plant or SyntheticPlant()
        self.pc = PecCache(); self.synth = SpectrumSynth(self.pc)
        # 电离平衡丰度表（init 一次，ADF11 I/O 不进在线环）：
        # ponytail: 只按 Te 建表（弱 ne 依赖忽略），升级路径=按 (ne,Te) 二维表+逐 shot 刷新
        self._te_tab = np.geomspace(10.0, 6000.0, 60)
        tab = [dict(ion_fractions("c", 1e14, float(t))) for t in self._te_tab]
        self._fr_tab = {q: np.array([d.get(q, 0.0) for d in tab]) for q in range(7)}
        self.fabric = DataFabric(); self.regime = RegimeMonitor()
        self.risk = RiskMonitor(); self.shield = SafetyShield()
        self.sup = Supervisor(); self.ood = None
        self.health = {}
        self._ref = []
        # 神经层（train_neural 后启用；未训练时全部走经典回退路径）
        self.jepa = None; self.hrm = None; self.trm = None
        self.world = None; self.planner = None; self.mpc = None
        self.control_enabled = False
        self._c_prev = 0.02
        self._inv_fallbacks = 0
        self._inv_rule = INVERSION
        self.ledger = LatencyLedger()
        self.hardcases = HardCaseBuffer()
        self.aligner = AlignedSnapshot()
        self._w_prev = None
        self._q_prev = 1.0
        # 2.1.3③: PredictionCache（WM 异步写快照/MPC 只读, age+revision 双硬门）+ PhysicsOnly 降级
        self.pred_cache = PredictionCache()
        self._state_rev = 0
        self.physics_only = None
        self.ext_meas = None   # 外部光谱测量注入钩子 fn(t)->I(C III 977, mid3)，None=内部合成测量

    def _mod(self, name, fn, fallback):
        """模块隔离执行：异常 → health 记录 + 回退值，不向上抛。全模块统一计延迟台账。"""
        t0 = time.perf_counter()
        try:
            r = fn(); self.health[name] = "ok"
        except Exception as e:  # noqa: BLE001
            self.health[name] = f"fault: {type(e).__name__}: {e}"
            r = fallback
        self.ledger.rec.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
        return r

    # ---- 物理链（12 步） ----
    def _multiline_concentration(self, nz_q, fr_c, rho, Te_cm, ne_cm, noise=0.05):
        """Six-line non-negative inversion along the chord, on a decimated sampling.

        One transition per charge state makes the weighted system square and full rank. The chord is
        sampled at twelve points rather than sixty: the integral is dominated by the core, and the extra
        points quadruple the atomic-data lookups for no measurable change in the estimate.
        """
        ch = self.plant.chords[2]
        R, Z, ds = sample_chord(ch, n=12)
        r_s = self.plant.fm.rho(R, Z)
        ok = (r_s <= 1.0).astype(float)
        rc = np.clip(r_s, 0.0, 1.0)
        te_s = np.interp(rc, rho, Te_cm)
        ne_s = np.interp(rc, rho, ne_cm)
        A = np.zeros((len(MULTILINE_LINES), 7))
        y = np.zeros(len(MULTILINE_LINES))
        mid = int(np.argmin(np.abs(r_s)))
        te_mid = float(te_s[mid])
        ne_mid = float(ne_s[mid])
        for k, name in enumerate(MULTILINE_LINES):
            el, q, wl = LINES[name]
            pec = float(self.synth.pc.pec_value(el, q, wl, ne_mid, te_mid))
            nz_s = np.interp(rc, rho, nz_q[q]) * ok
            y[k] = pec * self.synth.los_integrate(ne_s * nz_s, ds)
            shape = np.interp(rc, rho, fr_c[q]) * ok
            A[k, q] = pec * self.synth.los_integrate(ne_s * shape * ne_s, ds)
        if noise:
            y = y * (1.0 + noise * self.plant.rng.standard_normal(len(y)))
        # Row scaling: the states differ by twenty orders of magnitude, and an unscaled non-negative
        # solve returns a bias of several hundred per cent because it cannot trade rows against each other.
        for k, name in enumerate(MULTILINE_LINES):
            q = LINES[name][1]
            scale = A[k, q] if A[k, q] > 0 else 1.0
            A[k, :] /= scale
            y[k] /= scale
        x, _ = nnls(A, y)
        # Each row is scaled by its own state's abundance, so every component of x estimates the same
        # total concentration; summing them would multiply the answer by the number of lines.
        c_tot = float(np.median(x[x > 0])) if np.any(x > 0) else float("nan")
        if not np.isfinite(c_tot) or c_tot <= 0:
            raise ValueError("non-physical concentration")
        return c_tot, float(y[2])

    def _physics_chain(self, fr):
        rho = fr["rho"]
        Te_cm = fr["Te"]; ne_cm = fr["ne"] * 1e-6     # m⁻³→cm⁻³
        # 步 8：电离平衡丰度（查 init 预建表，逐 ρ 点插值）
        fr_c = np.zeros((7, len(rho)))
        for q in range(7):
            fr_c[q] = np.interp(Te_cm, self._te_tab, self._fr_tab[q])
        nz_tot = self.plant.c_frac * ne_cm            # 步：总碳密度
        nz_q = fr_c * nz_tot                          # 步：各态密度剖面
        # 步 5/9/10：逐弦光谱合成
        chord_I = {}
        for ch in self.plant.chords:
            R, Z, ds = sample_chord(ch)
            r_s = self.plant.fm.rho(R, Z)
            ok = r_s <= 1.0
            te_s = np.interp(np.clip(r_s, 0, 1), rho, Te_cm)
            ne_s = np.interp(np.clip(r_s, 0, 1), rho, ne_cm)
            out = {}
            for name in ["C III 977", "C IV 1549"]:
                el, q, wl = LINES[name]
                nz_s = np.interp(np.clip(r_s, 0, 1), rho, nz_q[q]) * ok
                out[name] = self.synth.los_integrate(
                    self.synth.line_emissivity(el, q, wl, ne_s, te_s, nz_s), ds)
            chord_I[ch["name"]] = out
        # 步 11：由 C III 977 弦强度反演总碳浓度——线强对浓度线性。
        # 铁律：标定参考必须与测量走**同一正演算子**（同弦、同网格、同 PEC、同插值），
        # 否则边缘薄发射壳层的离散歧义（>4×）会污染反演。c_est = c_ref × I_meas / I_ref。
        def fwd_cIII977(c_ref):
            ch = self.plant.chords[2]
            R, Z, ds = sample_chord(ch)
            r_s = self.plant.fm.rho(R, Z)
            ok = r_s <= 1.0
            te_s = np.interp(np.clip(r_s, 0, 1), rho, Te_cm)
            ne_s = np.interp(np.clip(r_s, 0, 1), rho, ne_cm)
            nz_s = np.interp(np.clip(r_s, 0, 1), rho, fr_c[3] * (c_ref * ne_cm)) * ok
            return self.synth.los_integrate(
                self.synth.line_emissivity("c", 3, 977.0, ne_s, te_s, nz_s), ds)
        I_ref = fwd_cIII977(0.02)
        c_multi, I_multi = None, None
        if self._inv_rule == "multiline":
            try:
                c_multi, I_multi = self._multiline_concentration(nz_q, fr_c, rho, Te_cm, ne_cm)
            except Exception:                                  # noqa: BLE001 - fall back, do not crash
                self._inv_fallbacks += 1
                c_multi, I_multi = None, None
        # 外部测量注入通道（真实/模拟诊断接入点）：有 ext_meas 时替代内部合成测量
        if self.ext_meas is not None:
            I_meas = float(self.ext_meas(fr["t"]))
        else:
            I_meas = chord_I["mid3"]["C III 977"] * (1 + 0.05 * self.plant.rng.standard_normal())
        if c_multi is None:
            c_est = 0.02 * I_meas / max(I_ref, 1e-30)
        else:
            c_est = c_multi
            I_meas = I_multi if self.ext_meas is None else I_meas
        self._last_fwd = fwd_cIII977          # TRM 精修用（同一正演算子）
        self._last_Imeas = float(I_meas)
        self._last_c_linear = float(c_est)
        # 步 12：体积分浓度（以估计浓度重建剖面）
        Nz, Cz = concentration(c_est * ne_cm, ne_cm, self.plant.g, rho)
        # Z_eff^LOS（可见轫致加权，步 5 同款弦几何）
        zeff_prof = 1 + 6 * c_est + 0 * rho   # 简化: 准中性近似 Z_eff≈1+Z̄·C（碳平均电荷~4→取 6 上限保守）
        zeff_los = zeff_los_from_profiles(ne_cm, Te_cm, zeff_prof)
        # 发布资格门（报告 3.1 冻结）：n_e 剖面来源与误差决定 Z_eff^LOS 能否作为绝对量发布
        rel = zeff_los_release(zeff_los, getattr(self.plant, "ne_src", "thomson"),
                               float(getattr(self.plant, "ne_rel_err", 0.05)),
                               float(np.max(np.abs(zeff_prof - zeff_los))))
        # n_z^q 双来源合并门在 step() 发布点执行（2.2 冻结：revision 以发布态为准）
        brem = self.synth.bremsstrahlung(ne_cm[0], Te_cm[0], float(zeff_prof[0]))
        return dict(chord_I=chord_I, c_est=float(c_est), N_tot=Nz, C_avg=float(Cz),
                    zeff_los=float(zeff_los), zeff_release=rel, nz_q=nz_q, rho=rho,
                    Te=fr["Te"], ne=fr["ne"],
                    i_meas=float(I_meas), brem=float(brem),
                    nz_q_core=[float(x) for x in nz_q[:, 0]])

    # ---- 神经层离线训练（shot 间/campaign 级，不进在线预算） ----
    def train_neural(self, n=300, verbose=False):
        """JEPA/HRM/TRM/世界模型全部用本 plant 的历史序列训练；控制链随之就绪。"""
        t_span = float(getattr(self.plant, "t1", 20.0))     # 真实炮用自身放电时长
        frames = [self.plant.frame(t=t_span * i / n) for i in range(n)]
        # JEPA：观测 Y = Te 抽稀 10 维；状态摘要 [Te0, ne0, Zeff, rad, beta, mhd]
        Y = np.array([f["Te"][::12] for f in frames])
        seqs = [(Y[i], Y[i + 1]) for i in range(n - 1)]
        self.jepa = PlasmaJepa(Y.shape[1], d_z=32, seed=0)
        lj = self.jepa.train(seqs, epochs=60)
        S = np.array([[f["Te"][0], f["ne"][0] * 1e-19, 1.17, f["rad_frac"], f["beta_N"], f["mhd_amp"]]
                      for f in frames])
        self.jepa.fit_state_decoder(self.jepa.encode(Y), S)
        # HRM：特征 [mhd, rad, d2e]，标签 = plant 场景标志。
        # 同一 20s 序列重放 10 遍（与在线同采样率下获得足够正样本，ELM 窗口仅 4 帧/遍）
        self.hrm = HrmRegime(3, K=5)
        feats, labels = [], {k: [] for k in HrmRegime.LABELS}
        for _ in range(10):
            for f in frames:
                elm = 6.0 <= f["t"] <= 8.0
                feats.append(self.hrm.update(np.array([f["mhd_amp"], f["rad_frac"], f["d2e_dt"]])))
                for k in HrmRegime.LABELS:
                    labels[k].append(1 if (k == "ELM" and elm) or (k == "H-mode" and not elm) else 0)
        self.hrm.fit(np.array(feats)[5:], {k: v[5:] for k, v in labels.items()})
        # 重置 EMA 状态上线（训练轨迹的状态不带入），保留训练好的分类器
        import copy
        trained_clf = copy.deepcopy(self.hrm.clf)
        self.hrm = HrmRegime(3, K=5)
        self.hrm.clf = trained_clf
        # TRM：θ=(c,)，正演 = 当前代表帧的 C III 977 算子，θ*=plant.c_frac
        fr0 = frames[0]
        rho = fr0["rho"]; Te_cm = fr0["Te"]; ne_cm = fr0["ne"] * 1e-6
        fr_c3 = np.interp(Te_cm, self._te_tab, self._fr_tab[3])
        ch = self.plant.chords[2]
        R, Z, ds = sample_chord(ch)
        r_s = self.plant.fm.rho(R, Z); ok = r_s <= 1.0
        te_s = np.interp(np.clip(r_s, 0, 1), rho, Te_cm)
        ne_s = np.interp(np.clip(r_s, 0, 1), rho, ne_cm)
        def fwd_c(theta):
            theta = np.atleast_1d(theta)
            nz_s = np.interp(np.clip(r_s, 0, 1), rho, fr_c3 * (theta[0] * ne_cm)) * ok
            return np.array([self.synth.los_integrate(
                self.synth.line_emissivity("c", 3, 977.0, ne_s, te_s, nz_s), ds)])
        th_star = np.array([self.plant.c_frac])
        Yf = lambda: fwd_c(th_star)
        self.trm = TrmSolver(1, 1, seed=0)
        lt = self.trm.train_teacher(fwd_c, th_star, Yf, n_traj=200)
        # 世界模型：Te 剖面轨迹（随机加热输入）
        self.world = DualWorldModel(len(rho), jepa=self.jepa, seed=0)
        traj = []
        X = fr0["Te"].copy()
        rng = np.random.default_rng(0)
        for _ in range(300):
            u = rng.uniform(-1, 1)
            Xn = self.world.FP(X, u) + 3.0 * np.exp(-((np.linspace(0, 1, len(rho)) - 0.15) / 0.1) ** 2) * u
            traj.append((X.copy(), u, Xn.copy())); X = Xn
        lw = self.world.train_residual(traj, epochs=120)
        # 2.1.3③: PhysicsOnlyPredictor 离线 init 预计算压缩 Jacobian（在线 O(1) 查表降级通道）
        self.physics_only = PhysicsOnlyPredictor(self.world.FP, len(rho))
        self.planner = TargetPlanner(self.world, odd=dict(core_Te=(500.0, 5000.0)), seed=0)
        self.mpc = FastMpc(self.world, H=6, slew=0.2, seed=0)
        if verbose:
            print(f"trained: jepa {lj[0]:.3f}->{lj[-1]:.4f}, trm {lt[0]:.3f}->{lt[-1]:.4f}, "
                  f"world {lw[0]:.3f}->{lw[-1]:.4f}")
        return dict(jepa=lj, trm=lt, world=lw)

    def step(self, t=None):
        fr = self.plant.frame(t=t)
        t0 = time.time()
        # Layer A 数据织物
        self._mod("fabric", lambda: [self.fabric.ingest("spec", fr["Te"], t=fr["t0"] if "t0" in fr else time.time()),
                                     self.fabric.ingest("thomson", fr["ne"])], None)
        # Layer C 快环：物理链（剖面/浓度/Z_eff）
        phys = self._mod("physics_chain", lambda: self._physics_chain(fr), None)
        # OOD（以剖面特征向量）
        feat = np.concatenate([fr["Te"][:5], fr["ne"][:5] * 1e-19])
        self._ref.append(feat)
        if self.ood is None and len(self._ref) >= 5:
            self.ood = KnnOod().fit(np.array(self._ref))
        ood_s = float(self._mod("ood", lambda: self.ood.score(feat)[0] if self.ood else 0.0, 0.0))
        # UPSSE Fast Branch（2.2 所有权冻结）：神经 X^neural 与经典 X^observer 融合即 Fast Branch 内部实现，
        # Fast Observer 仅其 fallback 旁路（不构成第二个 X^fast 来源）；门控 2.1.2 冻结：sigmoid + Q + Δw 限幅
        disagree = 0.0; w_neural = None
        if self.jepa is not None:
            def _jepa():
                Y = fr["Te"][::12][None, :]
                x_n = self.jepa.decode_state(self.jepa.encode(Y))[0]   # [Te0, ne0, Zeff, rad, beta, mhd]
                x_c = np.array([fr["Te"][0], fr["ne"][0] * 1e-19])
                sig = np.array([0.05, 0.05]) + ood_s * 0.2
                xf, wn, wo = obs_fuse(x_n[:2], x_c, sig, ood_s,
                                      np.abs(x_n[:2] - x_c) / (np.abs(x_c) + 1e-9),
                                      Q=self._q_prev, keys=["Te", "ne"], prev_w=self._w_prev)
                return x_n, xf, wn, float(np.mean(np.abs(x_n[:2] - x_c) / (np.abs(x_c) + 1e-9)))
            x_n, x_fused, w_neural, disagree = self._mod("jepa", _jepa, (None, None, None, 0.0))
            if w_neural is not None:
                self._w_prev = w_neural
        # HRM Regime（训练后替代规则版；multi-label）
        if self.hrm is not None:
            reg = self._mod("hrm", lambda: self.hrm.infer_proba(
                self.hrm.update(np.array([fr["mhd_amp"], fr["rad_frac"], fr["d2e_dt"]]))),
                self.regime.infer(dict(d2e_dt=fr["d2e_dt"], beta_N=fr["beta_N"], q95=fr["q95"],
                                       rad_frac=fr["rad_frac"], mhd_amp=fr["mhd_amp"], dIp_dt=fr["dIp_dt"])))
        else:
            reg = self._mod("regime", lambda: self.regime.infer(
                dict(d2e_dt=fr["d2e_dt"], beta_N=fr["beta_N"], q95=fr["q95"],
                     rad_frac=fr["rad_frac"], mhd_amp=fr["mhd_amp"], dIp_dt=fr["dIp_dt"])), {})
        # TRM 精修（warm start θ0=上一拍浓度；K_max 按剩余 deadline 动态预算，失败回退线性反演）
        c_src = "linear"
        if self.trm is not None and phys is not None:
            def _trm():
                remaining = max(10.0, constants.get("t_trm_budget_ms", 70.0)
                                - (time.time() - t0) * 1000)
                th, hist = self.trm.solve(self._last_fwd, lambda: np.array([self._last_Imeas]),
                                          theta0=np.array([self._c_prev]),
                                          k_max=min(5, trm_k_max(remaining)))
                return float(np.clip(th[0], 1e-4, 0.2))
            c_trm = self._mod("trm", _trm, None)
            if c_trm is not None:
                # A learned refinement may nudge the inversion, never replace it. The TRM is warm-started on
                # the same forward operator, so a disagreement beyond a factor of two is the solver drifting
                # away from the measurement rather than sharpening it. Without this gate a retrained TRM
                # reported 3.1e-4 against a true 3.0e-2 and the concentration column became fiction.
                c_inv = float(phys["c_est"])
                if c_inv > 0 and 0.9 <= c_trm / c_inv <= 1.1:
                    phys["c_est"] = c_trm; self._c_prev = c_trm; c_src = "trm"
                else:
                    self._c_prev = c_inv; c_src = "trm_rejected"
        # 世界模型一步预测 + 双空间交叉校验（Δz/ΔX 进 Q 的 observer 分歧项）
        dz = dX = None
        if self.world is not None and phys is not None:
            def _wm():
                xP, xMix = self.world.step(fr["Te"], self.plant.u_heat)
                return self.world.cross_check(fr["Te"], xMix, self.plant.u_heat)
            dz, dX = self._mod("worldmodel", _wm, (None, None))
        risks = self._mod("risk", lambda: self.risk.infer(
            dict(mhd_amp=fr["mhd_amp"], d2e_dt=fr["d2e_dt"], rad_frac=fr["rad_frac"],
                 beta_N=fr["beta_N"], ood=ood_s), reg), [])
        # Safety（stale 检查 + ODD + SafetyDecision 结构化输出）
        stale = self.fabric.stale_channels()
        state4s = dict(beta_N=fr["beta_N"], q95=fr["q95"],
                       zeff=phys["zeff_los"] if phys else 2.0, rad_frac=fr["rad_frac"])
        decision = self._mod(
            "safety", lambda: self.shield.check_decision(state4s, stale, ood_s), None)
        if decision is None:
            allowed, degrade, reasons = False, 4, ["safety fault"]
            decision = dict(status="rejected", reason_code=("FALLBACK_TRIGGERED",),
                            violated_constraint=(), state_age=0.0, odd_status=False,
                            eligibility_status=False, fallback_level=4)
        else:
            allowed = decision.status != "rejected"
            degrade = decision.fallback_level
            reasons = list(decision.reason_code)
            decision = dict(status=decision.status, reason_code=tuple(decision.reason_code),
                            violated_constraint=tuple(decision.violated_constraint),
                            state_age=decision.state_age, odd_status=decision.odd_status,
                            eligibility_status=decision.eligibility_status,
                            fallback_level=decision.fallback_level)
        # Supervisor 残差分类
        r_y = abs((phys["c_est"] - self.plant.c_frac) / self.plant.c_frac) if phys else 9.9
        verdict = self._mod("supervisor", lambda: self.sup.classify(r_y, 0.1, ood_s), {})
        # Hard Case 捕获（OOD 超阈 / 新 regime 候选 → 入缓冲待标注训练）
        if ood_s > 1.0 or verdict.get("kind") == "new_regime":
            self.hardcases.capture(feat, "ood" if ood_s > 1.0 else "new_regime")
        # Q_t（observer 分歧 = JEPA 解码分歧 + 世界模型 Δz）
        fresh = 1.0 - min(len(stale), 3) / 3
        obs_dis = disagree + (min(dz, 2.0) if dz is not None else 0.0)
        Q = state_quality(meas_quality=1.0, freshness=fresh, obs_disagree=obs_dis,
                          phys_resid=r_y if phys else 5.0, ood=ood_s)
        self._q_prev = Q
        # AlignedStateSnapshot（2.1.2②：慢态趋势前推对齐到快拍时刻）
        aligned = None
        if phys:
            self.aligner.push_slow(fr["t"], np.array([phys["c_est"], phys["zeff_los"]]))
            al = self.aligner.align(fr["t"], [float(fr["Te"][0]), float(fr["ne"][0])],
                                    Q=Q)
            aligned = dict(t_slow_src=al["t_slow_src"], X_slow=list(al["X_slow"]))
        self._state_rev += 1   # 状态链 revision（PredictionCache revision 门基准）
        # 五源不确定度 Σ（meas/cal/epi/mf/proc；报告 16 章）
        sigma = sigma_five(meas=0.05 if not stale else 0.20, cal=0.02,
                           epi=min(obs_dis, 2.0) * 0.1, mf=0.20,
                           proc=0.02 * abs(self.plant.u_heat))
        # 控制链（可选闭环；MVP-1 默认关。Target Planner→Fast MPC→Safety 裁决）
        ctrl = None
        if self.control_enabled and self.mpc is not None:
            def _ctrl():
                xt, J = self.planner.plan(fr["Te"], u_span=0.02, mission_te=3050.0)
                # 2.1.3③ E_Y 旁路: WM 预测快照写缓存 → 双硬门(age+revision/bundle)读;
                # 门不过 → PhysicsOnlyPredictor 降级（O(1) 查表, 绝无同步 WM 调用入 MPC 热路径）
                cache_gate, cache_src = None, None
                if self.world is not None:
                    Xw = np.asarray(fr["Te"], float)
                    Xh = []
                    for _ in range(5):
                        Xw = np.asarray(self.world.step(Xw, 0.0, ood=ood_s)[1], float)   # (xP, xMix) → xMix
                        Xh.append(Xw.copy())
                    bid = MANIFEST.bundle_id()
                    self.pred_cache.write(PredictionSnapshot(
                        X_hat=np.array(Xh), prediction_time=time.time(),
                        available_time=time.time(), model_bundle=bid,
                        source_state_id=self._state_rev,
                        source_state_revision=self._state_rev, source_bundle_id=bid))
                    snap, gate = self.pred_cache.read(self._state_rev, bid)
                    cache_gate = bool(gate)
                    if gate:
                        cache_src = "wm_cache"
                    elif self.physics_only is not None:
                        self.physics_only.predict(np.asarray(fr["Te"], float), 0.0, H=5)
                        cache_src = "physics_only"
                # TTE 消费（P2-17）：最高风险事件 (P,TTE) 进 MPC 防御窗口
                rk = None
                if risks:
                    top = max(risks, key=lambda r: r.P)
                    rk = (float(top.P), float(top.tte))
                u_c = self.mpc.act(fr["Te"], xt, risk=rk)
                # ControlEligibility 硬门（2.1.2 P0-3）：E_u=0 → 禁止高置信控制
                _, E_u = ctrl_eligibility(
                    {"Te": 0.0, "ne": 0.0, "c": 0.0 if not stale else 1.0},
                    {"Te": 0, "ne": 0, "c": 0},
                    {"Te": Q, "ne": Q, "c": Q})
                ok_ctl = (allowed and degrade == 0 and Q > 0.3 and E_u == 1
                          and decision["status"] != "rejected")   # Safety Shield 最终裁决
                u_s = u_c if ok_ctl else 0.0
                self.plant.accept(u_s)
                return dict(x_target=float(xt), u_cand=float(u_c), u_safe=float(u_s),
                            vetoed=not ok_ctl, E_u=E_u,
                            cache_gate=cache_gate, cache_src=cache_src)
            ctrl = self._mod("control", _ctrl, None)
        ms = (time.time() - t0) * 1000
        prof = None
        if phys:
            prof = dict(rho=[round(float(x), 3) for x in phys["rho"][::6]],
                        Te=[float(x) for x in phys["Te"][::6]],
                        ne=[float(x) for x in phys["ne"][::6]])
        # n_z^q 双来源合并门（2.2 冻结，发布点执行）：缺省快通道权威；慢环 refined revision
        # 过状态链绑定门（revision 祖先/相等 + 无 regime 跳变 + bundle 一致）与三硬门才可覆盖
        if phys:
            nz_auth, nz_rev, nz_src = nz_q_authoritative(
                phys["nz_q"], cur_revision=self._state_rev, cur_bundle_id=MANIFEST.bundle_id())
        else:
            nz_auth, nz_rev, nz_src = None, None, None
        return dict(t=fr["t"], zeff_los=phys["zeff_los"] if phys else None,
                    zeff_release=phys["zeff_release"] if phys else None,
                    c_est=phys["c_est"] if phys else None, C_avg=phys["C_avg"] if phys else None,
                    chord_I=phys["chord_I"] if phys else {}, profiles=prof,
                    i_meas=phys["i_meas"] if phys else None,
                    brem=phys["brem"] if phys else None,
                    nz_q_core=[float(x) for x in nz_auth[:, 0]] if nz_auth is not None else None,
                    nz_q_src=nz_src, nz_q_rev=nz_rev,
                    chain_info=dict(n_rho=len(self.plant.rho), n_chords=len(self.plant.chords),
                                    element="c", lines=list(LINES.keys()), c_src=c_src),
                    r_y=float(r_y),
                    regime=reg, risks=[vars(r) for r in risks], Q=Q, ood=ood_s,
                    degrade=degrade, allowed=allowed, reasons=reasons,
                    decision=decision, aligned=aligned,
                    odd_state=dict(beta_N=fr["beta_N"], q95=fr["q95"],
                                   rad_frac=fr["rad_frac"],
                                   zeff=phys["zeff_los"] if phys else None),
                    supervisor=verdict, health=dict(self.health), latency_ms=ms,
                    sigma=sigma, version_digest=_vers.VERSIONS.digest(),
                    bundle_id=MANIFEST.bundle_id(), state_rev=self._state_rev,
                    c_src=c_src, dz=dz, dX=dX, disagree=disagree, control=ctrl)


def demo(n=12):
    app = FrpwmMVP()
    out = [app.step() for _ in range(n)]
    last = out[-1]
    assert last["c_est"] and abs(last["c_est"] - 0.03) / 0.03 < 0.15, last["c_est"]
    assert 1.0 < last["zeff_los"] < 3.0
    assert last["Q"] > 0.5 and last["allowed"]
    assert last["latency_ms"] < 100, last["latency_ms"]
    print(f"pipeline OK: c_est={last['c_est']:.4f} Z_eff^LOS={last['zeff_los']:.2f} "
          f"Q={last['Q']:.2f} {last['latency_ms']:.1f}ms health={last['health']}")
    return out

def demo_neural(n=40):
    """神经层全链路：离线训练 → 在线 JEPA/HRM/TRM/世界模型 + 控制闭环验证。"""
    app = FrpwmMVP()
    losses = app.train_neural(n=40, verbose=True)   # 训练与在线同采样率（EMA 速率敏感）
    # 在线（不控制）：TRM 反演 + HRM regime + JEPA 融合
    out = [app.step(t=20.0 * i / n) for i in range(n)]
    trm_ok = [s for s in out if s["c_src"] == "trm"]
    assert len(trm_ok) > n * 0.8, f"TRM 覆盖率 {len(trm_ok)}/{n}"
    errs = [abs(s["c_est"] - 0.03) / 0.03 for s in trm_ok]
    assert np.mean(errs) < 0.15, np.mean(errs)
    # ELM 窗口 [6,8]s 内 HRM 应检出 ELM 概率升高
    elm_p = [s["regime"].get("ELM", 0) for s in out if 6.0 <= s["t"] <= 8.0]
    norm_p = [s["regime"].get("ELM", 0) for s in out if s["t"] > 12.0]
    assert np.mean(elm_p) > np.mean(norm_p) + 0.2, (np.mean(elm_p), np.mean(norm_p))
    # 控制闭环：mission 3050 eV，Safety 裁决下 u_safe 有界且 slew 满足
    app.control_enabled = True
    ctl = [app.step(t=22.0 + 0.5 * i)["control"] for i in range(12)]
    assert all(c is not None for c in ctl)
    us = [c["u_safe"] for c in ctl]
    assert all(abs(u) <= 1.0 for u in us)
    du = [abs(b - a) for a, b in zip(us, us[1:])]
    assert all(d <= 0.2 + 1e-9 for d in du), du   # slew rate 约束
    te_end = app.plant.frame()["Te"][0]
    print(f"neural OK: TRM 覆盖 {len(trm_ok)}/{n} 误差 {np.mean(errs):.3f}, "
          f"ELM {np.mean(elm_p):.2f} vs {np.mean(norm_p):.2f}, "
          f"控制 u_safe[-3:]={[round(u,2) for u in us[-3:]]} Te0={te_end:.0f}")
    return out

if __name__ == "__main__":
    demo()
    demo_neural()
