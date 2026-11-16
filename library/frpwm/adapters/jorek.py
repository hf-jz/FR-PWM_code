"""JOREK → FR-PWM 状态适配器（接口层；本身不产生任何 JOREK 数值）。

用途：把 JOREK 输出的非线性 MHD 状态（n_z, Te, ne, ψ 网格）接到本项目的观测算子与反演链上，
使"由独立代码生成的物理"可以作为估计器的检验集。这是论文 Stage 3/5 里 P0-2（正演与反演同算子）
唯一可落地的一半：数据来自另一个代码族，谱仍由本项目算子生成 → 独立的**输运/剖面**，非独立的
**原子/仪器**；论文口径必须照此写明。

HDF5 布局按 JOREK 文档 `docs/diagnostics/jorek_read_h5.md`：
  变量编号 0=psi, 1=u, 2=j, 3=w, 4=rho, 5=T（杂质模型另有 n_z 变量组）。
本模块只读、只映射，不做任何插值以外的假设；缺变量即报错，不静默补零。

用法:
    from frpwm.adapters.jorek import load_jorek_state
    st = load_jorek_state("/path/jorek_restart.h5")   # -> dict(rho, Te, ne, nz, R, Z, psi)
"""
import numpy as np

VAR = {0: "psi", 1: "u", 2: "j", 3: "w", 4: "rho", 5: "T", 6: "v_par"}


def _require(h5, keys, path=""):
    group = h5[path] if path else h5
    available = list(group.keys())
    missing = [k for k in keys if k not in available]
    if missing:
        raise KeyError(f"JOREK file missing {missing} — available: {available}")


# Real JOREK restart/output files carry no named datasets: the state lives in
# values(n_var, n_dof, n_tor, n_nodes) and coordinates in x(2, n_dof, n_tor, n_nodes).
# Variable order for model 199 is frozen in models/model199/mod_model_settings.f90
# (var_psi=1, var_u=2, var_zj=3, var_w=4, var_rho=5, var_T=6).
_IDX_MODEL199 = dict(psi=0, u=1, j=2, w=3, rho=4, T=5)


def load_jorek_values(path):
    """Read a **real** JOREK output (values/x blocks) and return per-node arrays.

    Returns dict(R, Z, psi, ne, Te, rho_psi) with one entry per node, using the
    model-199 variable order. Raises if the file is not a JOREK state file.
    Normalisation: JOREK stores normalised values (rho/rho_0, T/T_0); the
    reference factors are returned so the caller can restore physical units.
    """
    import h5py
    with h5py.File(path, "r") as f:
        if "values" not in f or "x" not in f:
            raise KeyError(f"not a JOREK state file (need 'values' and 'x'): {list(f.keys())[:8]}")
        values = np.asarray(f["values"][...], float)          # (n_var, n_dof, n_tor, n_nodes)
        xyz = np.asarray(f["x"][...], float)                  # (2, n_dof, n_tor, n_nodes)
        norm = {k: float(f[k][...].ravel()[0]) if k in f else None
                for k in ("sqrt_mu0_rho0", "sqrt_mu0_over_rho0", "central_density", "R_domm")}
    if values.ndim != 4:
        raise ValueError(f"unexpected 'values' rank {values.shape}")
    psi = values[_IDX_MODEL199["psi"], 0, 0, :]
    ne = values[_IDX_MODEL199["rho"], 0, 0, :]
    Te = values[_IDX_MODEL199["T"], 0, 0, :]
    scale = 1.0 / abs(norm["R_domm"] or 1.0)   # x/y are stored in units of R_domm
    R = xyz[0, 0, 0, :] * scale
    Z = xyz[1, 0, 0, :] * scale
    rho_psi = _psi_to_rho(psi)
    return dict(R=R, Z=Z, psi=psi, ne=ne, Te=Te, rho_psi=rho_psi, norm=norm)


def bin_to_flux_surface(R, Z, rho_psi, ne, Te, n_bins=32, rho_max=1.0):
    """Flux-surface average of per-node values onto n_bins in normalised psi.

    ponytail: plain arithmetic mean inside each rho bin (no |∇ψ| weighting);
    upgrade to volume weighting only if the comparison needs profile integrals.
    """
    edges = np.linspace(0.0, rho_max, n_bins + 1)
    rho_c = 0.5 * (edges[:-1] + edges[1:])
    ne_b = np.full(n_bins, np.nan)
    Te_b = np.full(n_bins, np.nan)
    R_b = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (rho_psi >= edges[i]) & (rho_psi < edges[i + 1])
        if not m.any():
            continue
        ne_b[i] = ne[m].mean()
        Te_b[i] = Te[m].mean()
        R_b[i] = R[m].mean()
    return dict(rho=rho_c, ne=ne_b, Te=Te_b, R_mid=R_b,
                n_nodes_in_bin=np.array([int(((rho_psi >= edges[i]) & (rho_psi < edges[i + 1])).sum())
                                         for i in range(n_bins)]))



def load_jorek_state(path, nz_key="n_z", require_fields=True):
    """读 JOREK restart/输出 HDF5 → 统一状态字典。

    返回 dict(rho, Te, ne, nz, R, Z, psi)，其中 rho 为归一化极向磁通（0..1），
    Te/ne/nz 为 (n_rho,) 或 (n_species, n_rho)。所有量保留文件里的物理单位（JOREK: m^-3, eV）。
    """
    import h5py
    with h5py.File(path, "r") as f:
        _require(f, ["psi", "T", "rho"], path="")
        psi = np.asarray(f["psi"][...], float)
        Te = np.asarray(f["T"][...], float)
        ne = np.asarray(f["rho"][...], float)          # JOREK 的 "rho" 是密度变量, 不是坐标
        nz = np.asarray(f[nz_key][...], float) if nz_key in f else None
        R = np.asarray(f["R"][...], float) if "R" in f else None
        Z = np.asarray(f["Z"][...], float) if "Z" in f else None
    if require_fields and nz is None:
        raise KeyError(f"'{nz_key}' absent: this file carries no impurity density, which is what the "
                       f"inversion needs; run a model that includes fluid impurities")
    rho = _psi_to_rho(psi)
    return dict(rho=rho, Te=_to_profile(Te), ne=_to_profile(ne),
                nz=None if nz is None else _to_profile(nz), R=R, Z=Z, psi=psi)


def _to_profile(a):
    """(n,) 或 (n_rho, n_species) → (n_rho,) 或 (n_species, n_rho)；单快照取末帧。"""
    a = np.asarray(a, float)
    if a.ndim == 3:                     # (n_frame, n_rho, n_species)
        a = a[-1]
    if a.ndim == 2:
        return a.T if a.shape[0] > a.shape[1] else a
    return a


def _psi_to_rho(psi):
    """归一化极向磁通（轴 0、LCFS 1）。JOREK 的径向索引从轴到边，故以首末点定标，
    不假设 ψ 的符号（JOREK 常取轴 0、边为负）。"""
    psi = np.asarray(psi, float)
    while psi.ndim > 1:                       # 单快照 → 1D
        psi = psi[0]
    axis, edge = float(psi[0]), float(psi[-1])
    if not np.isfinite(axis) or not np.isfinite(edge) or edge == axis:
        raise ValueError("degenerate psi map: cannot form a normalised flux coordinate")
    r = (psi - axis) / (edge - axis)
    if r[-1] < r[0]:                          # 守卫：径向方向必须朝外
        r = 1.0 - r
    return r


def write_fixture(path, n_rho=32, n_species=3, seed=7):
    """生成一个**合成** HDF5 夹具，仅用于自检接口形状（不是 JOREK 数据，勿用于论文）。"""
    import h5py
    rng = np.random.default_rng(seed)
    rho = np.linspace(0.0, 1.05, n_rho)
    psi = 1.0 - rho ** 2                                  # 单快照、单调 ψ 图
    Te = np.clip(800.0 * (1 - rho ** 1.5) + 20.0 + 5 * rng.normal(size=n_rho) * (1 - rho), 5.0, None)
    ne = np.clip(5e19 * (1 - rho ** 2) + 1e18, 1e17, None)
    nz = np.array([1e16 * (1 - rho ** 2) ** q for q in (1, 2, 3)])
    with h5py.File(path, "w") as f:
        f["psi"] = psi; f["T"] = Te; f["rho"] = ne; f["n_z"] = nz
        f["R"] = 1.7 + 0.5 * rho; f["Z"] = np.zeros_like(rho)
    return dict(path=path, n_rho=n_rho, n_species=n_species)


if __name__ == "__main__":
    import os, tempfile
    d = tempfile.mkdtemp(prefix="jorek_fixture_")
    fx = write_fixture(os.path.join(d, "fixture.h5"))
    st = load_jorek_state(fx["path"])
    assert st["rho"].shape == (fx["n_rho"],), st["rho"].shape
    assert st["nz"].shape == (fx["n_species"], fx["n_rho"]), st["nz"].shape
    assert abs(st["rho"].min()) < 1e-9 and abs(st["rho"].max() - 1.0) < 1e-9, (st["rho"].min(), st["rho"].max())
    assert np.all(np.diff(st["rho"]) > 0), "rho must be monotone for the interpolation downstream"
    print("jorek adapter OK:", {k: (None if v is None else np.shape(v)) for k, v in st.items()})
