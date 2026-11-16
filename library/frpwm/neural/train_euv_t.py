"""时变 PINN 联合训练（P2-⑩）：输入帧序列 (T,2048) → n_z(t,ρ) + θ=(dmax,decay,v,c)。
L_physics 时变：∂n/∂t + (1/V')∂(V'Γ)/∂r ≈ 0（预测 n_z(t) 时间差分 + 预测 θ 的 D/V 通量散度）。
用法: /usr/bin/python3 -m frpwm.neural.train_euv_t <data.npz> [epochs]"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

from frpwm.neural.euv_net import SpectralEncoder

_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "results", "euv_train")


class EUVNetT(nn.Module):
    """逐帧共享编码器 + 时间聚合(均值+末帧+一阶差分均值) → 序列头 n_z(t) 与 θ。"""

    def __init__(self, n_rho=32, n_pix=2048, z_dim=128, n_theta=4, n_t=8,
                 theta_mu=None, theta_sd=None, nz_mu=None):
        super().__init__()
        self.enc = SpectralEncoder(n_pix, z_dim)
        agg = z_dim * 3
        self.head_nz = nn.Sequential(nn.Linear(agg, 128), nn.GELU(), nn.Linear(128, n_t * n_rho))
        self.head_th = nn.Sequential(nn.Linear(agg, 64), nn.GELU(), nn.Linear(64, n_theta))
        self.n_t, self.n_rho = n_t, n_rho
        self.register_buffer("th_mu", torch.zeros(n_theta) if theta_mu is None else torch.tensor(theta_mu, dtype=torch.float32))
        self.register_buffer("th_sd", torch.ones(n_theta) if theta_sd is None else torch.tensor(theta_sd, dtype=torch.float32))
        self.register_buffer("nz_mu", torch.zeros(n_rho) if nz_mu is None else torch.tensor(nz_mu, dtype=torch.float32))

    def forward(self, x):                        # x (B, T, P)
        B, T, P = x.shape
        z = self.enc(x.reshape(B * T, P)).reshape(B, T, -1)
        agg = torch.cat([z.mean(1), z[:, -1], (z[:, 1:] - z[:, :-1]).mean(1)], dim=1)
        nz_log = self.head_nz(agg).reshape(B, T, self.n_rho) + self.nz_mu
        th = self.head_th(agg) * self.th_sd + self.th_mu
        return nz_log, th


def transport_residual_t(nz, theta, rho, ts, a_cm=60.0):
    """时变稳态残差: ∂n/∂t + (1/r)∂(rΓ)/∂r ≈ 0, Γ=-D∂n/∂r+Vn, D=dmax(1+decay·ρ²), V=-v·ρ。"""
    dmax = theta[:, 0:1]; decay = theta[:, 1:2]; v = theta[:, 2:3]
    D = dmax * (1.0 + decay * rho ** 2)
    V = -v * rho
    dr = (rho[1] - rho[0]) * a_cm
    dn = (nz[:, :, 1:] - nz[:, :, :-1]) / dr                    # (B,T,K-1)
    n_mid = 0.5 * (nz[:, :, 1:] + nz[:, :, :-1])
    D_mid = 0.5 * (D[:, None, 1:] + D[:, None, :-1])
    V_mid = 0.5 * (V[:, None, 1:] + V[:, None, :-1])
    gamma = -D_mid * dn + V_mid * n_mid
    r_mid = (0.5 * (rho[1:] + rho[:-1]) * a_cm)[None, None, :]
    div = ((r_mid * gamma)[:, :, 1:] - (r_mid * gamma)[:, :, :-1]) / (r_mid[:, :, 1:] * dr)
    dt = (ts[1:] - ts[:-1]).clamp_min(1e-6)[None, :, None]
    dn_dt = (nz[:, 1:, 1:-1] - nz[:, :-1, 1:-1]) / dt           # (B,T-1,K-2)
    res = dn_dt + div[:, 1:, :]
    scale = nz.mean(dim=(1, 2), keepdim=True) + 1e-30
    return ((res / scale) ** 2).mean()


def main(npz_path, epochs=600, seed=0, lam=(1.0, 0.5, 0.01, 0.0)):
    torch.manual_seed(seed); np.random.seed(seed)
    d = np.load(npz_path)
    frames = torch.tensor(d["frames"], dtype=torch.float32)     # (N,T,P)
    nz = torch.tensor(d["nz"], dtype=torch.float32)             # (N,T,K)
    params = torch.tensor(d["params"], dtype=torch.float32)
    params[:, 3] = torch.log10(params[:, 3])
    rho = torch.tensor(d["rho"], dtype=torch.float32)
    ts = torch.tensor(d["ts"], dtype=torch.float32)
    N, T, P = frames.shape
    X = torch.log1p(frames)
    X = (X - X.mean()) / (X.std() + 1e-9)
    nz_log = torch.log10(nz + 1e6)
    nz_mu = nz_log.mean((0, 1), keepdim=False)
    nz_sd = nz_log.std() + 1e-9
    th_mu, th_sd = params.mean(0), params.std(0) + 1e-9
    idx = np.random.permutation(N)
    n_tr = int(0.8 * N)
    tr, te = idx[:n_tr], idx[n_tr:]
    net = EUVNetT(n_rho=nz.shape[2], n_pix=P, n_t=T,
                  theta_mu=th_mu.numpy(), theta_sd=th_sd.numpy(),
                  nz_mu=nz_mu.numpy())
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.time()
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        nz_lp, th_p = net(X[tr])
        nz_p = (10 ** nz_lp - 1e6).clamp_min(0)
        l_data = ((nz_lp - nz_mu) / nz_sd - (nz_log[tr] - nz_mu) / nz_sd).pow(2).mean() \
            + ((th_p - params[tr]) / th_sd).pow(2).mean()
        # L_spec: 预测 n_z(t) 经 A 重建帧序列（高斯基矩阵预计算, 单次 matmul）
        if not hasattr(main, "_G"):
            sig = 0.15 / 2.355
            wl_grid = torch.tensor(d["wl_grid"], dtype=torch.float32)
            wls = torch.tensor(d["wls"], dtype=torch.float32)
            main._G = torch.stack([torch.exp(-0.5 * ((wl_grid - w) / sig) ** 2)
                                   / (sig * np.sqrt(2 * np.pi)) for w in wls])   # (L, P)
        A = torch.tensor(d["A"], dtype=torch.float32)[tr]       # (B,L,K)
        I_pred = torch.einsum("blk,btk->btl", A, nz_p)
        rec = torch.einsum("btl,lp->btp", I_pred, main._G)
        l_spec = (torch.log1p(rec) - torch.log1p(frames[tr])).pow(2).mean()
        l_phys = transport_residual_t(nz_p, th_p, rho, ts)
        loss = lam[0] * l_data + lam[1] * l_spec + lam[2] * l_phys
        loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                nz_lt, th_t = net(X[te])
                r2 = 1 - ((nz_lt - nz_log[te]) ** 2).mean() / nz_log[te].var()
                th_err = ((th_t - params[te]).abs() / (params[te].abs() + 1e-9)).mean(0)
            print(f"ep{ep}: loss={loss.item():.4f} data={l_data.item():.4f} spec={l_spec.item():.4f} "
                  f"phys={l_phys.item():.4f} | test R²={r2.item():.3f} "
                  f"θ%={[round(100*float(e),1) for e in th_err]}", flush=True)
    os.makedirs(_OUT, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(_OUT, "euv_net_t.pt"))
    rep = dict(epochs=epochs, n_train=int(len(tr)), n_test=int(len(te)),
               test_r2=float(r2.item()), theta_err_pct=[float(100 * e) for e in th_err],
               time_s=round(time.time() - t0, 1))
    json.dump(rep, open(os.path.join(_OUT, "train_report_t.json"), "w"), ensure_ascii=False, indent=1)
    print("train_t OK:", rep)
    return rep


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600)
