"""Z_eff^LOS 反演（MVP-1 第一输出量）：可见轫致弦平均 + 形状因子 ζ。"""
import numpy as np
from .. import constants

def zeff_los_from_profiles(ne_s, te_s, zeff_s):
    """正演：沿弦加权 Z_eff^LOS = ∫W n_e² Z dl/∫W n_e² dl，W=T_e^{-1/2}（可见轫致权重）。

    n_e² 加权是物理定义：轫致发射率 ε∝n_e²Z_eff·W，故弦积分亮度反解出的
    Z_eff 必然是 n_e² 加权弦平均（与同文件 zeff_from_brem 及 SpectrumSynth.bremsstrahlung
    的 n_e² 前向严格同算子；用 n_e 一次加权会在 n_e 剖面陡峭时产生系统性偏差）。
    """
    W = 1.0 / np.sqrt(np.maximum(te_s, 1.0))
    num = np.trapz(W * ne_s**2 * zeff_s)
    den = np.trapz(W * ne_s**2)
    return num / max(den, 1e-30)

NE_SRC_TRUSTED = ("thomson", "tomography", "abel")
NE_SRC_NONE = ("", "none", None)

def zeff_los_release(zeff_los, ne_src, ne_rel_err, delta_zeff, sigma0=0.0, mode_ok="measured"):
    """Z_eff^LOS 发布资格门（报告 3.1 冻结）：绝对 Z_eff 仅当 n_e 剖面来自
    Thomson/≥3 弦层析/Abel 且 δn_e/n_e ≤ (δn_e/n_e)_max；否则只能发布相对量。
    Σ 按一阶变分上界 |δZ| ≤ 2ΔZ·(δn_e/n_e) 膨胀（系统项，保守）。
    来源三态：可信（thomson/tomography/abel）→ 可绝对发布；有剖面但不可信 → imputed；
    无剖面 → fallback。→ dict(absolute, estimation_mode, sigma, allowed_in_safety, reason)
    """
    thr = float(constants.get("ne_profile_rel_max", 0.10))
    ne_rel_err = float(max(ne_rel_err, 0.0))
    delta_zeff = float(abs(delta_zeff))
    sigma = float(sigma0) + 2.0 * delta_zeff * ne_rel_err
    trusted = ne_src in NE_SRC_TRUSTED
    if trusted and ne_rel_err <= thr:
        return dict(absolute=True, estimation_mode=mode_ok, sigma=sigma,
                    allowed_in_safety=True, reason="ne_profile_ok")
    missing = ne_src in NE_SRC_NONE
    mode = "fallback" if missing else "imputed"        # 无剖面→fallback；有但不可信/超阈→imputed
    return dict(absolute=False, estimation_mode=mode, sigma=sigma, allowed_in_safety=False,
                reason="ne_profile_missing" if missing else "ne_profile_unconstrained")

def zeff_from_brem(I_brem, ne_los, te_los, wl_ang=5320.0):
    """反演：I_brem ∝ ∫n_e²·Z_eff·g·exp(-hc/λTe)/√Te dl → 解 Z_eff^LOS，再乘 ζ 修正。
    简化：用弦平均 n_e,Te 估算权重积分，I_brem 与正演同一任意单位体系。"""
    HC = 6.626e-34 * 3.0e8 * 1e10 / 1.602e-19
    gff = 1.2
    w = np.exp(-HC / (wl_ang * np.maximum(te_los, 1.0))) / np.sqrt(np.maximum(te_los, 1.0))
    z = I_brem / max(np.trapz(ne_los**2 * gff * w), 1e-30)
    return z * constants.get("shape_zeta", 1.51) / 1.51   # ζ 已在权重近似中归一

if __name__ == "__main__":
    s = np.linspace(0, 1, 200)
    ne = 5e19 * (1 - s**2) ** 0.3 + 2e18
    te = 3000 * (1 - s**2) ** 0.7 + 50
    ze = np.full(200, 1.8)
    z1 = zeff_los_from_profiles(ne, te, ze)
    assert abs(z1 - 1.8) < 0.02, z1
    # n_e² 加权回归检查：非均匀 Z_eff 剖面下必须与 n_e² 加权解析式一致，
    # 且与「n_e 一次加权」的结果可区分（后者是本轮修掉的物理错误）。
    ze2 = 1.2 + 0.8 * s  # 芯部低、边缘高
    W = 1.0 / np.sqrt(te)
    z_n2 = np.trapz(W * ne**2 * ze2) / np.trapz(W * ne**2)
    z_n1 = np.trapz(W * ne * ze2) / np.trapz(W * ne)
    zc = zeff_los_from_profiles(ne, te, ze2)
    assert abs(zc - z_n2) < 1e-12, (zc, z_n2)
    assert abs(z_n2 - z_n1) > 1e-3, "退化剖面无法区分 n_e / n_e² 加权"
    # 发布资格门（3.1 冻结）
    ok = zeff_los_release(zc, "thomson", 0.05, 0.5)
    assert ok["absolute"] and ok["allowed_in_safety"] and ok["estimation_mode"] == "measured", ok
    deg = zeff_los_release(zc, "thomson", 0.25, 0.5)          # 剖面不可信 → 相对量
    assert (not deg["absolute"]) and deg["estimation_mode"] == "imputed" and not deg["allowed_in_safety"], deg
    unt = zeff_los_release(zc, "imputed", 0.05, 0.5)          # 有剖面但不可信来源 → imputed
    assert (not unt["absolute"]) and unt["estimation_mode"] == "imputed", unt
    mis = zeff_los_release(zc, "none", 0.0, 0.5)              # 无剖面 → fallback
    assert (not mis["absolute"]) and mis["estimation_mode"] == "fallback" and not mis["allowed_in_safety"], mis
    assert deg["sigma"] > ok["sigma"] > 0.0                   # Σ 随 δn_e/n_e 膨胀
    print("zeff_los OK:", z1, "n_e^2-check:", zc, "vs n_e-weighted:", z_n1,
          "| release:", ok["reason"], deg["reason"], unt["reason"], mis["reason"])
