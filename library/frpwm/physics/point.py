# -*- coding: utf-8 -*-
"""POINT 极化干涉：磁面约束密度反演（main.pdf 诊断实例节，plan_eval09151210）。

冻结语义：
  y_POINT = [N_1..N_11, ψ_F1..ψ_F11]（11 干涉线积分 + 11 Faraday 极化，22 维观测向量）
  H_Y^(POINT)：Park-matrix 正演 N_i = Σ_j A_ij·n_e,j，A_ij = 弦 i 穿过磁面 bin j 的路径长度
  （几何由平衡 ψ(R,Z) 给出——n_e=n_e(ψ) 磁面降维：二维 10⁴ 未知 → 一维 ~20 磁面 bin）
  Faraday：ψ_F,i = C·λ²∫_{L_i} n_e B_∥ dl（内束 EFIT，改善 j_φ/q 剖面，本模块预留接口）
  反演：argmin_n ‖N_meas−A·n‖²_W + λ_s‖Dn‖²（平滑先验），n≥0——联合反演目标即
  冻结的 argmin_{X∈𝒞_t} 𝒥_t 的 L_obs(POINT)+L_obs(磁)+L_reg 实例（Grad–Shafranov 约束入 𝒫）
"""
import numpy as np


def park_matrix(fmap, chords, n_flux=20, n_samp=200):
    """Park 矩阵 A[i,j] = 弦 i 在磁面 bin j 内的路径长度（m）。
    fmap: FluxMap（rho(R,Z)）；chords: [{R0,Z0,dR,dZ,L}]；n_flux: 磁面分层数。"""
    from frpwm.physics.chords import sample_chord
    edges = np.linspace(0, 1, n_flux + 1)
    A = np.zeros((len(chords), n_flux))
    for i, ch in enumerate(chords):
        R, Z, ds = sample_chord(ch, n_samp)
        rho = fmap.rho(R, Z)
        j = np.clip(np.digitize(rho, edges) - 1, 0, n_flux - 1)
        inside = rho <= 1.0
        np.add.at(A[i], j[inside], ds)
    return A


def point_forward(A, n_e):
    """N_pred = A @ n_e（11 条线积分预测）。"""
    return A @ np.asarray(n_e, float)


def point_invert(N_meas, A, lam=0.3, w=None):
    """平滑正则最小二乘反演 n_e(ψ)：min ‖W(N−An)‖²+λ‖Dn‖²，n≥0（截断）。"""
    N_meas = np.asarray(N_meas, float)
    W = np.eye(A.shape[0]) if w is None else np.diag(w)
    D = np.diff(np.eye(A.shape[1]), axis=0)
    G = A.T @ W @ A + lam * (D.T @ D)
    n = np.linalg.solve(G, A.T @ W @ N_meas)
    return np.clip(n, 0, None)


def faraday_forward(A, n_e, bpar_mean):
    """ψ_F,i ∝ λ²∫ n_e B_∥ dl（预留：B_∥(弦,磁面) 由平衡/EFIT 给出，此处给平均场近似）。"""
    return np.asarray(bpar_mean, float) * (A @ np.asarray(n_e, float))


if __name__ == "__main__":
    from frpwm.physics.gfile import generate_gfile
    from frpwm.physics.fluxmap import FluxMap
    from frpwm.physics.chords import default_chords
    fmap = FluxMap(generate_gfile())
    # 合成 11 弦扇形（占位几何，Device Freeze 时换 DeviceConfig 实测弦参数）：
    # 5 水平 + 3 斜穿芯部 + 3 边缘切向——覆盖 ρ 全区间，避免重复弦退化
    R0, a = 1.7, 0.6
    chords = ([dict(name=f"h{i}", R0=R0 - a + 0.2 + (2 * a - 0.4) * i / 4, Z0=0.0, dR=0.0, dZ=1.0, L=2.5) for i in range(5)]
              + [dict(name=f"d{i}", R0=R0 - 0.3 + 0.3 * i, Z0=-1.2, dR=0.1 * (i - 1), dZ=1.0, L=2.4) for i in range(3)]
              + [dict(name=f"t{i}", R0=R0 + a - 0.15, Z0=-0.3 + 0.3 * i, dR=0.0, dZ=1.0, L=0.9) for i in range(3)])
    A = park_matrix(fmap, chords, n_flux=20)
    assert A.shape == (11, 20) and (A >= 0).all() and A.sum() > 0
    rho20 = (np.arange(20) + 0.5) / 20
    n_true = 7.98e13 * (1 - rho20 ** 2)
    N = point_forward(A, n_true)
    n_est = point_invert(N, A, lam=0.3)
    rel = np.abs(n_est - n_true).mean() / n_true.mean()
    assert rel < 0.15, rel                       # 11 弦 vs 20 bin（平滑先验） honest 界
    psi_f = faraday_forward(A, n_true, 2.0)
    hit = A.sum(1) > 0
    assert psi_f.shape == (11,) and (psi_f[hit] > 0).all() and hit.sum() >= 8   # 边缘切向弦可空（占位几何）
    print(f"PASS point: Park A(11,20), 反演 rel={rel:.3f}<0.15, Faraday 接口 OK（{hit.sum()}/11 弦命中）")
