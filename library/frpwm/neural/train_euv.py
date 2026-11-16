"""PINN 约束联合训练管线（P2-⑨）：L = λ1·L_data + λ2·L_spec + λ3·L_physics + λ4·L_boundary。
  L_data   : log n_z 剖面 + θ 参数监督（STRAHL 库标签）
  L_spec   : ‖log1p(A@n_z_pred) − log1p(frame)‖²（A=同算子发射矩阵, 可微正演）
  L_physics: 稳态输运残差（euv_net.transport_residual）
  L_boundary: 边缘密度惩罚（n_z(ρ=1) 应小）
用法: /usr/bin/python3 -m frpwm.neural.train_euv <data.npz> [epochs]"""
import json, os, sys, time
import numpy as np
import torch

from frpwm.neural.euv_net import EUVNet, transport_residual

_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "results", "euv_train")


def main(npz_path, epochs=300, seed=0, lam=(1.0, 0.5, 0.01, 0.05)):
    torch.manual_seed(seed); np.random.seed(seed)
    d = np.load(npz_path)
    frames = torch.tensor(d["frames"], dtype=torch.float32)
    nz = torch.tensor(d["nz"], dtype=torch.float32)
    params = torch.tensor(d["params"], dtype=torch.float32)
    params[:, 3] = torch.log10(params[:, 3])          # c_frac 对数化
    A = torch.tensor(d["A"], dtype=torch.float32)
    rho = torch.tensor(d["rho"], dtype=torch.float32)
    n = len(frames)
    X = torch.log1p(frames)
    X = (X - X.mean()) / (X.std() + 1e-9)
    nz_log = torch.log10(nz + 1e6)                    # 物理下限 1e6 m? cm^-3（避免 0 值 -30 地板支配损失）
    nz_mu, nz_sd = nz_log.mean(0, keepdim=True), nz_log.std(0, keepdim=True) + 1e-9
    th_mu, th_sd = params.mean(0), params.std(0) + 1e-9
    idx = np.random.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    net = EUVNet(n_rho=nz.shape[1], n_pix=frames.shape[1],
                 theta_mu=th_mu.numpy(), theta_sd=th_sd.numpy(),
                 nz_mu=nz_mu.squeeze(0).numpy())
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.time()
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        nz_log_p, th_p = net(X[tr])
        nz_p = (10 ** nz_log_p - 1e6).clamp_min(0)
        l_data = ((nz_log_p - nz_mu) / nz_sd - (nz_log[tr] - nz_mu) / nz_sd).pow(2).mean() \
            + ((th_p - params[tr]) / th_sd).pow(2).mean()
        I_pred = torch.bmm(A[tr], nz_p.unsqueeze(-1)).squeeze(-1)      # 线强 (B, n_lines)
        frame_rec = torch.zeros_like(frames[tr])
        # 帧域 L_spec：用线强加权重建（与 frame_labels 同高斯公式, 但微分路径走 I_pred）
        sig = 0.35 / 2.355
        wl_grid = torch.tensor(d["wl_grid"], dtype=torch.float32)
        wls = torch.tensor(d["wls"], dtype=torch.float32)
        for li, wl0 in enumerate(wls):
            frame_rec += I_pred[:, li:li + 1] / (sig * np.sqrt(2 * np.pi)) * \
                torch.exp(-0.5 * ((wl_grid - wl0) / sig) ** 2)
        l_spec = (torch.log1p(frame_rec) - torch.log1p(frames[tr])).pow(2).mean()
        l_phys = transport_residual(nz_p, th_p, rho)
        l_bnd = (10 ** nz_log_p[:, -1]).pow(2).mean() / ((10 ** nz_log_p).mean() ** 2 + 1e-30)
        loss = lam[0] * l_data + lam[1] * l_spec + lam[2] * l_phys + lam[3] * l_bnd
        loss.backward(); opt.step()
        if ep % 60 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                nz_lt, th_t = net(X[te])
                r2 = 1 - ((nz_lt - nz_log[te]) ** 2).mean() / nz_log[te].var()
                th_err = ((th_t - params[te]).abs() / (params[te].abs() + 1e-9)).mean(0)
            print(f"ep{ep}: loss={loss.item():.4f} data={l_data.item():.4f} spec={l_spec.item():.5f} "
                  f"phys={l_phys.item():.4f} bnd={l_bnd.item():.4f} | test R²(log n_z)={r2.item():.3f} "
                  f"θ误差%={[round(100*e,1) for e in th_err.tolist()]}", flush=True)
    os.makedirs(_OUT, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(_OUT, "euv_net.pt"))
    rep = dict(epochs=epochs, n_train=int(len(tr)), n_test=int(len(te)),
               test_r2=float(r2.item()), theta_err_pct=[float(100 * e) for e in th_err],
               time_s=round(time.time() - t0, 1))
    json.dump(rep, open(os.path.join(_OUT, "train_report.json"), "w"), ensure_ascii=False, indent=1)
    print("train OK:", rep)
    return rep


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
