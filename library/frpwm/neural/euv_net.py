"""全帧谱编码器 + 多头解码器（torch, CPU 可训）。
SpectralEncoder: 1D conv ×3(stride4) → TransformerEncoder(2 层) → mean-pool → z
EUVNet: z → n_z(ρ) 头(softplus 保正) + θ=(dmax,decay,v) 头
联合损失见 train_euv.py：L_data + L_spec(A@n_z 微分正演) + L_physics(稳态输运残差) + L_boundary。"""
import numpy as np
import torch
import torch.nn as nn


class SpectralEncoder(nn.Module):
    def __init__(self, n_pix=2048, z_dim=128, d_model=64, nhead=4, nlayer=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(32, d_model, 9, stride=4, padding=4), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                           batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, nlayer)
        self.proj = nn.Linear(d_model, z_dim)

    def forward(self, x):                    # x (B, n_pix) log1p 归一化帧
        h = self.conv(x.unsqueeze(1))        # (B, d, L)
        h = self.tr(h.transpose(1, 2))       # (B, L, d)
        return self.proj(h.mean(1))          # (B, z)


class EUVNet(nn.Module):
    def __init__(self, n_rho=32, n_pix=2048, z_dim=128, n_theta=4,
                 theta_mu=None, theta_sd=None, nz_mu=None):
        super().__init__()
        self.enc = SpectralEncoder(n_pix, z_dim)
        self.head_nz = nn.Sequential(nn.Linear(z_dim, 128), nn.GELU(), nn.Linear(128, n_rho))
        nn.init.zeros_(self.head_nz[-1].weight); nn.init.zeros_(self.head_nz[-1].bias)
        self.head_th = nn.Sequential(nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, n_theta))
        self.register_buffer("th_mu", torch.zeros(n_theta) if theta_mu is None else torch.tensor(theta_mu, dtype=torch.float32))
        self.register_buffer("th_sd", torch.ones(n_theta) if theta_sd is None else torch.tensor(theta_sd, dtype=torch.float32))
        self.register_buffer("nz_mu", torch.zeros(n_rho) if nz_mu is None else torch.tensor(nz_mu, dtype=torch.float32))

    def forward(self, x):
        z = self.enc(x)
        nz_log = self.head_nz(z) + self.nz_mu                       # 零初始化末层 → 起点=均值预测
        th = self.head_th(z) * self.th_sd + self.th_mu
        return nz_log, th


def transport_residual(nz, theta, rho, a_cm=60.0):
    """L_physics: 稳态磁面平均输运残差 (1/V')d(V'Γ)/dρ ≈ 0（芯部源≈0）,
    Γ = -D·dn/dr + V·n, D(ρ)=dmax·(1+decay·ρ²), V(ρ)=-v·ρ (cm/s→SI 由数据尺度吸收)。
    可微（torch 有限差分）。注: strahl 内部 funct 形式为近似——正则项非精确等式。"""
    dmax = theta[:, 0:1]; decay = theta[:, 1:2]; v = theta[:, 2:3]
    D = dmax * (1.0 + decay * rho ** 2)              # (B, K)
    V = -v * rho                                     # 内向为负
    dr = (rho[1] - rho[0]) * a_cm
    dn = (nz[:, 1:] - nz[:, :-1]) / dr               # (B, K-1)
    n_mid = 0.5 * (nz[:, 1:] + nz[:, :-1])
    D_mid = 0.5 * (D[:, 1:] + D[:, :-1])
    V_mid = 0.5 * (V[:, 1:] + V[:, :-1])
    gamma = -D_mid * dn + V_mid * n_mid              # 通量 (B, K-1)
    r_mid = 0.5 * (rho[1:] + rho[:-1]) * a_cm
    vp = torch.clamp(r_mid, min=1.0)                 # V'∝r（圆截面近似）
    div = (vp * gamma)[:, 1:] - (vp * gamma)[:, :-1]
    scale = (nz.mean(dim=1, keepdim=True) + 1e-30)
    return ((div / scale) ** 2).mean()


if __name__ == "__main__":
    x = torch.randn(4, 2048)
    net = EUVNet()
    nz_log, th = net(x)
    assert nz_log.shape == (4, 32) and th.shape == (4, 4)
    rho = torch.linspace(0, 1, 32)
    nz = (10 ** nz_log - 1e6).clamp_min(0)
    r = transport_residual(nz, th, rho)
    assert r.item() >= 0
    n_params = sum(p.numel() for p in net.parameters())
    print(f"euv_net OK: nz_log{nz_log.shape}, θ{th.shape}, L_physics={r.item():.3e}, params={n_params/1e3:.0f}k")
