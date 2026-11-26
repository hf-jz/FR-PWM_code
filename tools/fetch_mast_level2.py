"""Fetch the calibrated MAST charge-exchange products (zarr v3) as npz.

    python3 tools/fetch_mast_level2.py 28315 28316 ...     # default: 28315..28324

The public level-2 release is zarr v3 and the installed zarr is 2.x, so the arrays are read the
way that does not depend on the client version: fetch the single chunk object, `zstd -dc` it, and
parse the little-endian float64 buffer.  Each of these arrays is one unchunked block.

Shape comes from the byte count rather than from the group metadata: the object store serves the
787 kB consolidated metadata in ~73 s per shot and each array in ~30 s, so asking for the
metadata would dominate the transfer.  The major-radius axis is 161 in every shot of the
campaign, which fixes the frame count as nbytes/8/161; the check below refuses anything else.
"""
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://s3.echo.stfc.ac.uk/mast/level2/shots/%s.zarr"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "artefacts", "A22_SD_TIF", "inputs")
NR = 161                                   # major-radius nodes, constant across the campaign


def chunk(shot, key, ndim):
    """zarr v3 chunk key: c/<i> for 1-D, c/<i>/<j> for 2-D."""
    url = BASE % shot + "/charge_exchange/%s/c/%s" % (key, "/".join(["0"] * ndim))
    with urllib.request.urlopen(url, timeout=300) as r:
        raw = r.read()
    return subprocess.run(["zstd", "-dc"], input=raw, stdout=subprocess.PIPE, check=True).stdout


def array(shot, key):
    buf = chunk(shot, key, 1 if key in ("major_radius", "time") else 2)
    n = len(buf) // 8
    if key in ("major_radius", "time"):
        return np.frombuffer(buf, dtype="<f8")[:n]
    assert n % NR == 0, "%s: %d values is not a whole number of %d-node frames" % (key, n, NR)
    # the object is written row-major with the radius axis first: (161, n_time)
    return np.frombuffer(buf, dtype="<f8")[:n].reshape(NR, n // NR)


def one(shot):
    keys = ("major_radius", "time", "t_i", "v_i")
    with ThreadPoolExecutor(len(keys)) as ex:
        vals = list(ex.map(lambda k: array(shot, k), keys))
    out = dict(zip(keys, vals))
    assert np.all(np.diff(out["major_radius"]) > 0), "radius axis not monotone"
    assert out["t_i"].shape == (NR, out["time"].size), (out["t_i"].shape, out["time"].size)
    assert out["v_i"].shape == out["t_i"].shape
    out["shot"] = np.int64(shot)   # type: ignore[assignment]
    p = os.path.join(DEST, "level2_%s.npz" % shot)
    np.savez(p, **out)
    ti = out["t_i"]
    print("shot %s  t_i %s  finite %.2f  median %.0f eV" % (shot, ti.shape,
          np.mean(np.isfinite(ti)), np.nanmedian(ti)), flush=True)


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    for s in ([int(x) for x in sys.argv[1:]] or list(range(28315, 28325))):
        one(s)
