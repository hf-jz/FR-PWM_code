"""UPSSE — Unified Physics-Aware Spectral State Estimation（统一光谱物理反演主链）。

架构来源：intelligence_stack/docs/chatgpt/plan_eval09121744.md——把原先三条并列路线
（Zeff 网络 / H/D/T PINN / 杂质 Bayesian+TRM）统一为一条主链：
  Shared Spectral Encoder → Physics-aware Fusion（physics token + 轫致）
  → Species-specific Physics Heads（H/D/T 同位素头 / W,C 杂质头 / D,V 头 / n_z 头 / Zeff 快头）
  → Differentiable CR/Atomic/Balmer Layer（n_q→CR→PEC→I_pred，∂I/∂n_q 可反传）
  → Physics-informed Recursive State Estimator（TRM 循环精修：x→Physics→r=Y−Ŷ→Δx→x+αΔx）
  → Zeff_phys + Zeff_AI → bayes_fuse → Z_eff ± σ → （接 World Model / MPC）
统一损失：L = L_obs + λ1·L_CR + λ2·L_transport + λ3·L_isotope + λ4·L_consistency
            + λ5·L_uncertainty + λ6·L_recursive。
ponytail 口径：CR 速率/PEC 表是查表插值（对 Te/ne 不可微）——∂I/∂n_q 解析可微，
Te/ne 经 physics token 条件化进入编码器路径；Te/ne 的全可微代理属于后续升级。
"""
import numpy as np
import torch
import torch.nn as nn

from frpwm.neural.euv_net import SpectralEncoder, transport_residual  # 复用已验证组件
from frpwm.neural.pgm_zeff import bayes_fuse                          # 复用融合/不一致检测

# Balmer 同位素位移（Hα/Dα/Tα 近似线心, Å）
BALMER_L0 = (6562.8, 6561.0, 6559.9)


class UPSSE(nn.Module):
    """统一物理感知光谱状态估计网络。"""

    SOLVER_PROFILE_ID = "RecursiveSolverProfile-TRM-v1"   # plan_eval09140906 RPS 实现 profile 命名

    def __init__(self, n_pix=2048, n_q=75, n_rho=32, n_bal=128, d=96, n_phys=5, line_basis=None,
                 regime_dim=0):
        super().__init__()
        if line_basis is not None:
            # 物理感知识谱前端：固定参数去卷积（G 伪逆，65 线近正交 fwhm=0.15≈6σ 间隔，良态）——
            # 像素域证据（train_euv 同数据：65 线特征 R²=0.85 vs 原始像素 60 样本过拟合）
            self.register_buffer("Gpinv", torch.pinverse(
                torch.as_tensor(line_basis, dtype=torch.float32)))          # (2048,65)
            self.enc = nn.Sequential(nn.Linear(2 * line_basis.shape[0], 128), nn.GELU(),
                                     nn.Linear(128, 128), nn.GELU())        # 65 线强编码器（log+线性双标度）
        else:
            self.Gpinv = None
            self.enc = SpectralEncoder(n_pix, 128)               # 共享谱编码器（EUV 支路）
        self.bal_enc = nn.Sequential(nn.Linear(n_bal, 96), nn.GELU(), nn.Linear(96, 128),
                                     nn.GELU())              # 共享谱编码器（Visible/Balmer 支路）
        self.norm_e = nn.LayerNorm(128)                      # 支路输出归一——防 EUV 幅度淹没
        self.norm_b = nn.LayerNorm(128)                      # Balmer 弱信号支路（实测差 100×）
        self.phys_enc = nn.Sequential(nn.Linear(n_phys, 64), nn.GELU(), nn.Linear(64, d))
        self.brem_enc = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d))
        self.spec_proj = nn.Linear(128, d)
        self.fuse = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.0)
        # Species-specific Physics Heads（直读 [融合 z, 谱表示 h]——融合均值池化稀释
        # 支路信息（Balmer 在 3-token 均值中占 1/3），实测纯标签 800ep iso 仍 22% 的瓶颈）
        dh = d + 128
        self.head_hdt = nn.Linear(dh + 128, 4)               # X_H,X_D,X_T (softmax) + B (softplus)，+Balmer 支路特征
        self.head_fr = nn.Linear(dh, n_q)                    # W 电荷态分布 fr_q (log10)
        self.head_cw = nn.Sequential(nn.Linear(dh + d, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus())  # +物理 token 直读
        self.head_cc = nn.Sequential(nn.Linear(dh + d, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus())
        self.head_dv = nn.Linear(dh, 3)                      # (dmax, decay, v) 归一化
        self.head_nz = nn.Linear(dh, n_rho)                  # log10 n_z(ρ)
        self.head_z = nn.Linear(dh, 2)                       # Zeff 快头 μ, logσ
        # Recursive State Estimator（TRM 精修算子，阻尼 α 固定 0.5）
        self.alpha = 0.5
        self.refine = nn.Sequential(nn.Linear(d + 128, 192), nn.GELU(), nn.Linear(192, d + 1))
        # Recursive Physics Solver 功能形式（plan_eval09131124 冻结）：
        #   θ_{k+1} = θ_k + α_k·c_k·R_φ(θ_k, r_k, M_t)   —— 输出 (Δθ, logit c_k)，c_k=σ(·)∈(0,1)
        # M_t 通道：regime_dim>0 时启用（Regime Reasoning Module 上下文，默认实现见 neural/regime.py）
        self.mt_proj = nn.Linear(regime_dim, 128) if regime_dim > 0 else None
        self.register_buffer("nz_mu", torch.zeros(n_rho))
        self.register_buffer("qgrid", torch.arange(n_q, dtype=torch.float32))

    def heads(self, z, h, bf=None, pf=None):
        zh = torch.cat([z, h], 1)
        fr = torch.pow(10.0, self.head_fr(zh).clamp(-12, 0))
        fr = fr / fr.sum(1, keepdim=True)
        # hdt 头加读 Balmer 支路特征（EUV⊕Balmer 求和后 EUV 仍主导，同位素信息需直读）
        zh_h = torch.cat([z, h, bf], 1) if bf is not None else zh
        # cw/cc 头加读物理 token（P_RAD/ne 幅度在融合均值池化稀释 1/3；W-256 实测 cw 13-17%
        # 远高于 P_RAD 直读地板 3%）
        zh_c = torch.cat([zh, pf], 1) if pf is not None else zh
        hdt = self.head_hdt(zh_h)
        x_hdt = torch.softmax(hdt[:, :3], 1)                 # hard-param：和=1（L_isotope 结构实现）
        B = nn.functional.softplus(hdt[:, 3:4]) + 0.2
        q = self.qgrid
        cw = self.head_cw(zh_c).squeeze(-1) * 1e-4
        cc = self.head_cc(zh_c).squeeze(-1) * 1e-2
        z_phy = 1.0 + cw * ((q * q * fr).sum(1) - (q * fr).sum(1)) + 30.0 * cc
        zz = self.head_z(zh)
        return dict(z=z, h=h, bf=bf, pf=pf, fr=fr, x_hdt=x_hdt, B=B, cw=cw, cc=cc,
                    dv=self.head_dv(zh), nz_log=self.head_nz(zh) + self.nz_mu,
                    mu_ai=zz[:, 0], sig_ai=zz[:, 1].clamp(-6, 2).exp(),
                    z_phy=z_phy)

    def forward(self, frame, bal, phys, brem):
        if self.Gpinv is not None:
            li = (torch.expm1(frame) @ self.Gpinv).clamp_min(1e-3)
            x = torch.cat([torch.log1p(li), li * 1e-13], 1)    # log 域(剖面/nz)+线性域(幅度/cw)双标度
        else:
            x = frame
        e = self.norm_e(self.enc(x))
        bf = self.norm_b(self.bal_enc(bal))
        h = e + bf                                                 # EUV⊕Visible 共享谱表示
        pf = self.phys_enc(phys)
        tok = torch.stack([self.spec_proj(h), pf, self.brem_enc(brem)], 1)
        return self.heads(self.fuse(tok).mean(1), h, bf, pf)


def gauss_basis(wl_grid, centers, fwhm):
    """高斯基矩阵 (n_line, n_pix)，fwhm 为标量或逐线向量。
    面积归一（与数据集正演同算子 euv_frame.frame：I += inten/(sig√2π)·exp(−0.5((λ−λ0)/sig)²)，
    sig=fwhm/2.355——峰值归一会让像素幅度低 15.7×，log1p 域闭环系统性 ~40%）。"""
    c = torch.as_tensor(centers, dtype=torch.float32).reshape(-1, 1)
    g = torch.as_tensor(wl_grid, dtype=torch.float32).reshape(1, -1)
    fw = torch.as_tensor(fwhm, dtype=torch.float32)
    if fw.dim() == 0:
        fw = fw.expand(c.shape[0])
    fw = fw.reshape(-1, 1)
    sig = fw / 2.355
    return torch.exp(-4.0 * np.log(2.0) * ((g - c) / fw) ** 2) / (sig * float(np.sqrt(2 * np.pi)))


class AtomicLayer:
    """可微 CR/Atomic/Balmer 正演层：x → Ŷ（EUV 线帧 + Balmer 帧），全部 torch 可反传。
    EUV 正演 = 数据集前向矩阵 A（静态 Te/几何下逐案例恒定，实测 rel-std 1e-15）：
    y_lines = A @ n_z(ρ) —— 线强直接由 nz 头剖面给出（闭环 C1 与剖面精度 C3 物理耦合，
    替代旧 rank-1 幅度模型 c·k_l——后者无法表达输运参数驱动的线比变化，是 C1 58% 的根因）。"""

    def __init__(self, wl_grid, A_matrix, bal_grid, fwhm_euv=0.15, line_wl=None):
        self.register = None
        self.A = torch.as_tensor(A_matrix, dtype=torch.float32)          # (65, n_rho)
        self.G_euv = None
        if line_wl is not None:
            self.G_euv = gauss_basis(wl_grid, line_wl, fwhm_euv)          # (65, 2048)
        self.bal_grid = torch.as_tensor(bal_grid, dtype=torch.float32)

    def forward(self, o):
        nz = torch.pow(10.0, o["nz_log"])                                # (B, n_rho)
        y_lines = nz @ self.A.T                                          # (B, 65)
        y_euv = y_lines @ self.G_euv if self.G_euv is not None else y_lines
        g = self.bal_grid.reshape(1, 1, -1)
        c = torch.tensor(BALMER_L0).reshape(1, 3, 1)
        w = 0.35 * o["B"].reshape(-1, 1, 1)                              # (B,1,1) 逐样本展宽
        gb = torch.exp(-4.0 * np.log(2.0) * ((g - c) / w) ** 2)          # (B, 3, n_bal)
        y_bal = (o["x_hdt"].unsqueeze(-1) * gb).sum(1)                   # (B, n_bal)
        return y_euv, y_bal

    def residual(self, o, y_euv_obs, y_bal_obs, n_pix=2048):
        """观测空间残差（log1p 域与观测同尺度，抽稀到 128 维喂 TRM）。"""
        y1, y2 = self.forward(o)
        r = torch.cat([(torch.log1p(y1) - y_euv_obs)[:, :: n_pix // 96],
                       (y2 - y_bal_obs)[:, :: 2]], 1)              # 96+64→160 → 截 128
        return r[:, :128], (y1, y2)


def recursive_refine(model, layer, o0, y_euv, y_bal, K=3, M_t=None):
    """Physics-informed Recursive State Estimator（Recursive Physics Solver 默认实现，
    plan_eval09131124 功能形式冻结）：θ_{k+1}=θ_k+α·c_k·R_φ(θ_k,r_k,M_t)。
    M_t: Regime Reasoning 上下文 (B, regime_dim)，None 时按零上下文。"""
    outs = [o0]
    o = o0
    for _ in range(K):
        r, _ = layer.residual(o, y_euv, y_bal)
        if model.mt_proj is not None:
            mt = M_t if M_t is not None else torch.zeros(r.shape[0], model.mt_proj.in_features)
            r = r + model.mt_proj(mt)
        dzc = model.refine(torch.cat([o["z"], r], 1))
        dz, c_k = dzc[:, :-1], torch.sigmoid(dzc[:, -1:])
        o = model.heads(o["z"] + model.alpha * c_k * dz, o["h"], o["bf"], o["pf"])
        outs.append(o)
    return outs
