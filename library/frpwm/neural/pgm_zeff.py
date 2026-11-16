"""PGM-Zeff（Physics-Guided Multimodal Zeff Network）与 Baseline-A 对照实现。

架构来源：intelligence_stack/docs/chatgpt/plan_eval09112158.md 专家评估——
原 I(λ)→Normalization→Dense+PE→Conv1D→BiGRU→Transformer→Zeff 定位为 Baseline-A（强基线），
FR-PWM 最终方案 = SpectralTransformer + CR + Bremsstrahlung + ne/Te + PINN + Uncertainty：
  EUV/Visible 谱 → Physics-aware Spectral Transformer
                + EUV/Visible Bremsstrahlung + (ne,Te,Prad,Ip,Bt,...)
                → Multimodal Latent Plasma State
                → CR/Impurity State Head → {n_Wq+,n_C,n_O,...}
                → Physics-derived Zeff + Direct AI Zeff
                → Bayesian/Ensemble Fusion → Zeff ± σ。
关键点：Z_eff 不直接黑箱回归——同一 Z_eff 可对应完全不同的杂质组合与谱形（简并），
故网络输出杂质态，Z_eff 作为派生物理量；AI/物理双路径不一致度 d_Z 触发 OOD 降权。
"""
import numpy as np
import torch
import torch.nn as nn


def _pe(n, d):
    p = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    f = torch.exp(-np.log(10000.0) * torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(p * f)
    pe[:, 1::2] = torch.cos(p * f[: pe[:, 1::2].shape[1]])
    return pe


class BaselineA(nn.Module):
    """I(λ)→Norm→Dense+PE→Conv1D→BiGRU→Transformer→Zeff（原方案，对照基线）。"""

    def __init__(self, n_pix=2048, d=64):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.register_buffer("pe", _pe(n_pix, d))
        self.conv = nn.Sequential(nn.Conv1d(d, d, 9, stride=4, padding=4), nn.GELU(),
                                  nn.Conv1d(d, d, 9, stride=4, padding=4), nn.GELU())
        self.gru = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        layer = nn.TransformerEncoderLayer(d, 4, 128, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):                    # x (B, n_pix) log1p 帧
        h = self.inp(x.unsqueeze(-1)) + self.pe           # (B, L, d)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)  # (B, L/4, d)
        h, _ = self.gru(h)
        h = self.tr(h)
        return self.head(h.mean(1)).squeeze(-1)           # (B,)


class PGMZeff(nn.Module):
    """Physics-Guided Multimodal Zeff Network（FR-PWM 最终方案 D）。"""

    def __init__(self, n_pix=2048, d=96, n_q=75, n_phys=5):
        super().__init__()
        # Physics-aware Spectral Transformer：ConvStem 提局部峰形 → 谱 patch → Transformer
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, d, 9, stride=4, padding=4), nn.GELU())   # 2048 → 128 patches
        self.register_buffer("pe", _pe(n_pix // 16, d))
        layer = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, 2)
        # Physics Token 编码器（ne,Te,Prad,Ip,Bt,...）与 Bremsstrahlung 通道编码器
        self.phys_enc = nn.Sequential(nn.Linear(n_phys, 64), nn.GELU(), nn.Linear(64, d))
        self.brem_enc = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d))
        # 多模态融合：[谱, 物理, 轫致] 三 token cross-attention → latent plasma state
        self.fuse = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.0)
        # CR / Impurity State Head：电荷态分布 fr_q（log10）+ 浓度 c（W）与 c2（第二杂质 C）
        self.head_cr = nn.Linear(d, n_q)
        self.head_c = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus())
        self.head_c2 = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus())
        # Direct AI Zeff Head：μ, logσ
        self.head_z = nn.Linear(d, 2)
        self.register_buffer("qgrid", torch.arange(n_q, dtype=torch.float32))

    def forward(self, frame, phys, brem):
        h = self.stem(frame.unsqueeze(1)).transpose(1, 2) + self.pe   # (B, 128, d)
        z_spec = self.tr(h).mean(1)
        tok = torch.stack([z_spec, self.phys_enc(phys), self.brem_enc(brem)], dim=1)
        z = self.fuse(tok).mean(1)                                    # latent plasma state
        log_fr = self.head_cr(z)                                      # (B, n_q) log10 fr
        fr = torch.pow(10.0, log_fr.clamp(-12, 0))
        fr = fr / fr.sum(dim=1, keepdim=True)
        c = self.head_c(z).squeeze(-1) * 1e-4                         # (B,) W 浓度（物理标度）
        c2 = self.head_c2(z).squeeze(-1) * 1e-2                        # (B,) C 浓度（物理标度）
        zz = self.head_z(z_spec)                                      # Direct AI 头=纯谱路径
        mu_ai, log_sig = zz[:, 0], zz[:, 1].clamp(-6, 2)
        q = self.qgrid
        # Physics-derived Zeff：W 项（CR 头 fr 自洽）+ C 项（keV 下 C 全电离, Σq²fr−Σqfr=36−6=30）
        z_phy = 1.0 + c * ((q * q * fr).sum(1) - (q * fr).sum(1)) + 30.0 * c2
        return dict(mu_ai=mu_ai, sig_ai=log_sig.exp(), z_phy=z_phy,
                    fr=fr, c=c, c2=c2, z=z)


def bayes_fuse(mu_ai, sig_ai, z_phy, sig_phy, d_max=2.0):
    """Bayesian/Ensemble 融合 + AI/Physics disagreement 检测（专家 §10）。
    d_Z = |Z_AI−Z_PHY|/√(σ_AI²+σ_PHY²)；d_Z>d_max 触发不一致 → w_AI 降权。
    返回 (z_fused, sig_fused, d_Z, w_AI)。"""
    d_z = (mu_ai - z_phy).abs() / (sig_ai ** 2 + sig_phy ** 2).sqrt()
    w_ai = 1.0 / sig_ai ** 2
    w_phy = torch.as_tensor(1.0 / sig_phy ** 2, dtype=w_ai.dtype)
    w_ai = w_ai * torch.exp(-(d_z - d_max).clamp(min=0.0))            # 不一致降权
    z = (w_ai * mu_ai + w_phy * z_phy) / (w_ai + w_phy)
    sig = 1.0 / (w_ai + w_phy).sqrt()
    return z, sig, d_z, w_ai / (w_ai + w_phy)
