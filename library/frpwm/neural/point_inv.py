# -*- coding: utf-8 -*-
"""POINT NN 反演头（EAST 最新发展，plan_eval09151210 §二十/引[6]）：
神经网络直接从 11 通道线积分密度反演 n_e(ρ)——训练数据由平衡磁面（EFIT/gfile）+
Park-matrix 正演生成（同算子铁律：训练/在线同一 forward operator，train_w96 同款口径）。
定位：Fast Branch 反演头（1-10 ms）；point_invert（平滑 LSQ）作物理基线对照；
闭环语义：n_pred → point_forward → 与实测 N 比较 → 残差进 Recursive Physics Solver。
loss = 标签 MSE + 闭环 ‖A·n_pred − N‖²（physics-informed，非纯回归）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch import nn

from frpwm.physics.gfile import generate_gfile
from frpwm.physics.fluxmap import FluxMap
from frpwm.physics.point import park_matrix, point_invert


class PointInvNet(nn.Module):
    """N(11) → n_e(ρ, 20)：MLP 反演头（线性输出——Softplus 输出层实测梯度死亡
    （gnorm≈5e-4、out_sd=0，负预激活区 softplus'→0）；非负性由 eval 截断实现）。"""

    def __init__(self, n_ch=11, n_flux=20, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_ch, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, n_flux))

    def forward(self, N):
        return self.net(N)


def gen_dataset(A, n=400, seed=7, noise=0.02):
    """剖面族 n0(1−ρ^α)^β + 2% 测量噪声 → (N, n) 监督对（Park 正演生成）。"""
    rng = np.random.default_rng(seed)
    n_flux = A.shape[1]
    rho = (np.arange(n_flux) + 0.5) / n_flux
    al = rng.uniform(0.8, 3.0, n)
    be = rng.uniform(0.3, 2.5, n)
    n0 = rng.uniform(3e13, 1.2e14, n)
    prof = n0[:, None] * np.clip(1 - rho[None, :] ** al[:, None], 0, 1) ** be[:, None]
    N = (A[None, :, :] @ prof[:, :, None]).squeeze(-1)
    N_obs = N * (1 + noise * rng.standard_normal(N.shape))
    return N_obs.astype(np.float32), prof.astype(np.float32), rho


if __name__ == "__main__":
    torch.manual_seed(7)
    fmap = FluxMap(generate_gfile())
    R0, a = 1.7, 0.6
    chords = ([dict(name=f"h{i}", R0=R0 - a + 0.2 + (2 * a - 0.4) * i / 4, Z0=0.0, dR=0.0, dZ=1.0, L=2.5) for i in range(5)]
              + [dict(name=f"d{i}", R0=R0 - 0.3 + 0.3 * i, Z0=-1.2, dR=0.1 * (i - 1), dZ=1.0, L=2.4) for i in range(3)]
              + [dict(name=f"t{i}", R0=R0 + a - 0.15, Z0=-0.3 + 0.3 * i, dR=0.0, dZ=1.0, L=0.9) for i in range(3)])
    A = park_matrix(fmap, chords, n_flux=20)
    N, PROF, rho = gen_dataset(A, n=400)
    itr, ite = slice(0, 300), slice(300, 400)
    T = torch.as_tensor
    At = T(A.astype(np.float32))
    mu = T(PROF[itr].mean(0)) / 1e14
    model = PointInvNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500)
    for ep in range(1500):
        opt.zero_grad()
        pred = (model(T(N[itr] * 1e-16)) + mu) * 1e14     # 输入 1e16 归一，输出 1e14 归一
        l_lab = (((pred - T(PROF[itr])) / 1e14) ** 2).mean()
        l_clo = ((pred @ At.T - T(N[itr])) ** 2).mean() / (T(N[itr]) ** 2).mean()
        (l_lab + 0.5 * l_clo).backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = ((model(T(N[ite] * 1e-16)) + mu) * 1e14).clamp_min(0)   # 非负截断（eval）
        rel_nn = float((pred - T(PROF[ite])).abs().mean() / T(PROF[ite]).mean())
        clo_nn = float((((pred @ At.T - T(N[ite])) ** 2).mean() / (T(N[ite]) ** 2).mean()) ** 0.5)
    rels = []
    for i in range(300, 400):
        n_est = point_invert(N[i], A, lam=0.3)
        rels.append(np.abs(n_est - PROF[i]).mean() / PROF[i].mean())
    rel_lsq = float(np.mean(rels))
    print(f"NN rel={rel_nn:.4f} vs LSQ rel={rel_lsq:.4f} | NN 闭环 rel={clo_nn:.4f}")
    assert rel_nn < rel_lsq, "NN 反演应优于 2% 噪声下 LSQ 基线"
    assert clo_nn < 0.05, clo_nn
    print("PASS point_inv_net: NN 反演优于 LSQ 基线且闭环 <5%")
