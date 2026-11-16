"""真实 shot 数据接入（tokamark-v1 / FAIR-MAST Zarr）。
batch tar → 每 shot 一个 .zarr：读取 39 谐波信号中对 FR-PWM 有用的多诊断子集，
统一时间基准（不重采样，保留原生率）→ Y_t 序列 + provenance（附录 B Schema）。"""
import os, tarfile, zipfile, shutil
from .._env import external  # noqa: E402
import numpy as np
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import zarr

SIGNALS = {
    "ip": ("summary", "ip"),
    "ne": ("summary", "line_average_n_e"),
    "p_rad": ("summary", "power_radiated"),
    "greenwald": ("summary", "greenwald_density"),
    "dalpha": ("spectrometer_visible", "dalpha_channel"),
    "beta_N": ("equilibrium", "beta_normal"),
    "elong": ("equilibrium", "elongation"),
    "gas": ("gas_injection", "outboard_total"),
    "magn": ("magnetics", "i_plasma"),
}

def unpack_batch(tar_path, out_dir, max_shots=50):
    """batch tar → zips → zarr dirs（只取前 max_shots 个，避免 565GB 全解）。"""
    os.makedirs(out_dir, exist_ok=True)
    zips_dir = out_dir + "_zips"
    os.makedirs(zips_dir, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            tf.extract(m, zips_dir)
            zp = os.path.join(zips_dir, m.name)
            dest = os.path.join(out_dir, m.name[:-4])
            if not os.path.exists(dest):
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(dest)
            os.remove(zp)
            n += 1
            if n >= max_shots:
                break
    shutil.rmtree(zips_dir, ignore_errors=True)
    return sorted(os.listdir(out_dir))

def _zpath(p):
    """zip 解出常嵌套一层同名目录，取真正含 .zgroup 的那层。"""
    if os.path.exists(os.path.join(p, ".zgroup")) or os.path.exists(os.path.join(p, ".zmetadata")):
        return p
    subs = [os.path.join(p, d) for d in os.listdir(p)]
    subs = [s for s in subs if os.path.isdir(s)]
    return subs[0] if len(subs) == 1 else p

def load_shot(zpath, signals=None):
    """读一个 shot 的可用信号 → {key: (t, y)}（FAIR-MAST 组布局，各组自带 time）。"""
    signals = signals or SIGNALS
    root = zarr.open(_zpath(zpath), mode="r")
    out = {}
    for k, (grp, name) in signals.items():
        if grp not in root or name not in root[grp]:
            continue
        g = root[grp]
        try:
            y = np.asarray(g[name][:]).squeeze()
            t = np.asarray(g["time"][:]) if "time" in g else None
            y = y.astype(float); t = None if t is None else t.astype(float)
        except Exception:  # noqa: BLE001
            continue
        if y.ndim > 1:
            y = y[:, 0]
        if t is None or len(t) != len(y):
            t = np.arange(len(y)) * 1e-3
        ok = np.isfinite(y) & np.isfinite(t)
        if ok.sum() > 5:
            out[k] = (t[ok], y[ok])
    return out

def build_dataset(shot_dirs, max_shots=40):
    """多 shot → JEPA 训练序列：以 ne 的时间基准为主，其余信号插值对齐（训练专用，在线不重采样）。
    → (seqs[(Y_t,Y_{t+1})], states, meta)。"""
    seqs, states, meta = [], [], []
    KEYS = ("ip", "ne", "p_rad", "dalpha", "beta_N", "gas")
    for sd in shot_dirs[:max_shots]:
        try:
            d = load_shot(sd)
        except Exception:  # noqa: BLE001
            continue
        anchor = "ne" if "ne" in d else ("ip" if "ip" in d else None)
        if anchor is None:
            continue
        t = d[anchor][0]
        if len(t) < 20:
            continue
        cols = []
        for k in KEYS:
            if k not in d:
                cols.append(np.zeros((len(t), 1), np.float32)); continue
            tk, yk = d[k]
            cols.append(np.interp(t, tk, yk).reshape(-1, 1).astype(np.float32))
        Y = np.concatenate(cols, -1).astype(np.float32)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
        for i in range(len(Y) - 1):
            seqs.append((Y[i], Y[i + 1]))
            states.append(Y[i])
        meta.append(os.path.basename(sd))
    return seqs, np.array(states, np.float32), meta

if __name__ == "__main__":
    import sys
    tar = external("batch_00000.tar", "TOKAMARK_DIR", "directory holding the public MAST archive")
    out = "/tmp/mast_zarr"
    shots = unpack_batch(tar, out, max_shots=10)
    print("shots:", len(shots), shots[:3])
    d = load_shot(os.path.join(out, shots[0]))
    print("signals:", {k: v[1].shape for k, v in d.items()})
    seqs, S, meta = build_dataset([os.path.join(out, s) for s in shots])
    print("seqs:", len(seqs), "states:", S.shape, "meta:", len(meta))
    assert len(seqs) > 100
    print("mast OK")
