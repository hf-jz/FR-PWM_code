# -*- coding: utf-8 -*-
"""Zeff control 算子 A[·]（main.pdf P1-4/plan_eval09141034 冻结数学定义）。

Z_eff^control = A[Z_eff^profile(ρ)]，三种算子（DeviceConfig 选定其一）：
    A_core[Z]      = ∫₀^ρc w(ρ)Z(ρ)dρ / ∫₀^ρc w(ρ)dρ
    A_vol[Z]       = ∫₀¹ V'(ρ)Z(ρ)dρ / ∫₀¹ V'(ρ)dρ
    A_max,Ω[Z]     = max_{ρ∈Ω} Z(ρ)
离散实现用与网格一致的求积权重；同一 DeviceConfig 下各语言实现必须逐位一致。
"""
import numpy as np

OPS = ("CORE_AVG", "VOLUME_AVG", "REGION_MAX")


def a_core(z, rho, w=None, rho_c=0.3):
    """core average：ρ∈[0,ρc] 权重积分平均（w 默认 1，网格梯形求积）。"""
    z, rho = np.asarray(z, float), np.asarray(rho, float)
    m = rho <= rho_c
    w = np.ones_like(rho) if w is None else np.asarray(w, float)
    return float(np.trapezoid(w[m] * z[m], rho[m]) / np.trapezoid(w[m], rho[m]))


def a_vol(z, rho, vprime):
    """volume average：V'(ρ) 加权体积平均（grid-consistent 梯形求积）。"""
    z, rho, vprime = np.asarray(z, float), np.asarray(rho, float), np.asarray(vprime, float)
    return float(np.trapezoid(vprime * z, rho) / np.trapezoid(vprime, rho))


def a_region_max(z, rho, omega=(0.0, 1.0)):
    """指定区域 maximum：Ω=[lo,hi] 内 max。"""
    z, rho = np.asarray(z, float), np.asarray(rho, float)
    m = (rho >= omega[0]) & (rho <= omega[1])
    return float(z[m].max())


def zeff_control(z, rho, op, vprime=None, w=None, rho_c=0.3, omega=(0.0, 1.0)):
    """Z_eff^control = A[Z_eff^profile]；op ∈ OPS（DeviceConfig 指定）。"""
    assert op in OPS, op
    if op == "CORE_AVG":
        return a_core(z, rho, w, rho_c)
    if op == "VOLUME_AVG":
        assert vprime is not None, "VOLUME_AVG 需 V'(ρ)"
        return a_vol(z, rho, vprime)
    return a_region_max(z, rho, omega)


if __name__ == "__main__":
    rho = np.linspace(0, 1, 33)
    z = 1.5 + 0.5 * rho ** 2                     # 解析：core(ρc=1,w=1)=1.5+0.5/3；vol(V'=2ρ)=1.5+0.5*0.4
    assert abs(a_core(z, rho, rho_c=1.0) - (1.5 + 0.5 / 3)) < 5e-3
    assert abs(a_vol(z, rho, 2 * rho) - (1.5 + 0.5 * 0.5)) < 5e-3   # ∫2ρ(1.5+0.5ρ²)/∫2ρ = 1.5+0.25
    assert a_region_max(z, rho, (0.5, 1.0)) == float(z.max())
    assert zeff_control(z, rho, "CORE_AVG", rho_c=1.0) == a_core(z, rho, rho_c=1.0)
    print("PASS zeff_ops: A_core/A_vol/A_max 解析对拍一致")
