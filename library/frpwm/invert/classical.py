"""L1 Abel/半解析反演 与 L3 RF 代理——复用 spectra-real-time-system/rt 已验证实现。"""
import sys
from .._env import external  # noqa: E402
sys.path.insert(0, external("", "SPEC2CONC", "path of the plasma_spec2conc package"))
import invert_l1 as _l1      # noqa: F401  半解析核 + 正则化 Abel（0.099ms/帧）
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import l3_surrogate as _l3   # noqa: F401  RF 蒸馏代理（2.3ms, L1×L3 逆方差融合）
