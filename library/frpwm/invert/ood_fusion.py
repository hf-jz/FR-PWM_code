"""多证据 OOD 融合（P0-2，报告 16/17）。
2.1.3 证据尺度校准：五证据原始量纲不同，先逐证据 Calibration_i→[0,1] 经验尾概率再加权；
缺失证据权重重归一化；相关证据（Ensemble/Reconstruction 共享模型组）组内取 max 防重复计入。"""
import numpy as np
from .. import constants

class MultiEvidenceOod:
    def __init__(self, knn=None):
        self.knn = knn
        self.w = dict(constants.get("ood_weights"))
        self._mu = None; self._prec = None
        self._run = 0
        self._ref_scores = None                        # 各证据在参考集上的原始分（校准用）
    def fit(self, X_ref):
        self._mu = X_ref.mean(0)
        cov = np.cov(X_ref.T) + 1e-6 * np.eye(X_ref.shape[1])
        self._prec = np.linalg.inv(cov)
        if self.knn is not None and getattr(self.knn, "ref", None) is None:
            self.knn.fit(X_ref)
        # 校准集：参考集逐样本各证据原始分（经验 CDF 的分布底座，勿批均化成标量）
        self._ref_scores = {}
        for x_i in X_ref:
            for k, v in self.scores(x_i).items():
                if v is not None:
                    self._ref_scores.setdefault(k, []).append(v)
        self._ref_scores = {k: np.sort(np.asarray(v)) for k, v in self._ref_scores.items()}
        return self
    def scores(self, x, sigma_ens=None, recon=None, phys=None):
        """x: 样本（批）。未提供的证据返回 None（缺失，融合时重归一化，不按 0 计入）。"""
        x = np.atleast_2d(np.asarray(x, float))
        d = x - self._mu
        s_knn = float(self.knn.score(x)[0]) if self.knn is not None else None
        s_mah = float(np.sqrt(np.maximum((d @ self._prec * d).sum(-1), 0)).mean() / 3.0)
        return dict(knn=s_knn, mah=s_mah,
                    ens=None if sigma_ens is None else float(sigma_ens),
                    rec=None if recon is None else float(recon),
                    phys=None if phys is None else float(phys))
    def calibrate(self, scores):
        """逐证据概率尺度校准：s_i → s̃_i = 参考集经验 CDF（该分数在历史数据中的极端程度，
        域内→~0.5，域外→~1.0）。缺失证据（None）保持缺失。退化参考集（常数）按超界二值。"""
        if self._ref_scores is None:
            return dict(scores)
        out = {}
        for k, v in scores.items():
            if v is None:
                out[k] = None; continue
            ref = self._ref_scores.get(k)
            if ref is None or ref.size == 0:
                out[k] = float(np.clip(v, 0, 1)); continue
            if ref[-1] <= ref[0]:                          # 退化（全同值）：超界即极端
                out[k] = 1.0 if v > ref[-1] else 0.0
            else:
                out[k] = float(np.searchsorted(ref, v, side="right") / ref.size)
        return out
    def fuse(self, scores_cal):
        """S_OOD = Σ w_i·s̃_i（校准后同一概率尺度）。
        缺失证据（None）按可用集合重归一化；ens/rec 相关证据组内取 max、权重合并。"""
        avail = {k: v for k, v in scores_cal.items() if v is not None}
        if not avail:
            return 1.0                                    # 全部缺失 → 保守判域外
        # 相关证据组：ensemble 与 reconstruction 共享模型组，取 max、权重按实际在场合并
        if "ens" in avail or "rec" in avail:
            w_pair = (self.w.get("ens", 0.2) if "ens" in avail else 0.0) + \
                     (self.w.get("rec", 0.2) if "rec" in avail else 0.0)
            s_pair = max(avail.get("ens", 0.0), avail.get("rec", 0.0))
            parts = [(w_pair, s_pair)]
            parts += [(self.w.get(k, 0.2), v) for k, v in avail.items() if k not in ("ens", "rec")]
        else:
            parts = [(self.w.get(k, 0.2), v) for k, v in avail.items()]
        w_sum = sum(w for w, _ in parts)
        return float(sum(w * s for w, s in parts) / max(w_sum, 1e-30))   # 缺失重归一
    def evaluate(self, x, **kw):
        sc_raw = self.scores(x, **kw)
        sc = self.calibrate(sc_raw)
        S = self.fuse(sc)
        self._run = self._run + 1 if S > 0.99 else 0
        return S, sc, self._run >= constants.get("n_hyst", 5)

if __name__ == "__main__":
    from .ood import KnnOod
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (400, 6))
    od = MultiEvidenceOod(KnnOod()).fit(ref)
    S_in, _, trig_in = od.evaluate(rng.normal(0, 1, (5, 6)))
    S_out = 0.0
    for _ in range(6):
        S_out, sc_out, trig_out = od.evaluate(rng.normal(8, 1, (5, 6)))
    assert S_out > S_in and trig_out, (S_in, S_out)
    assert 0.0 <= S_in <= 1.0 and 0.0 <= S_out <= 1.0                    # 概率尺度
    assert S_out > 0.99, S_out                                           # 域外尾概率≈1
    # 缺失证据重归一：phys=None 不应拉低 S
    sc_miss = dict(sc_out); sc_miss["phys"] = None
    S_miss = od.fuse(sc_miss)
    assert S_miss >= S_out - 1e-9, (S_miss, S_out)
    # 相关证据组内 max：ens=0.9/rec=0.1 不按 0.5 平均
    S_grp = od.fuse({"knn": 0.0, "mah": 0.0, "ens": 0.9, "rec": 0.1, "phys": 0.0})
    assert abs(S_grp - 0.4 * 0.9) < 1e-9, S_grp
    print(f"ood_fusion OK: S_in={S_in:.3f} S_out={S_out:.3f} 滞回={trig_out} 缺失重归一/组内max 过")
