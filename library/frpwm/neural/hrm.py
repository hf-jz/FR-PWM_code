"""HRM Regime Reasoning（报告 11.4：HRM 唯一职责 = regime 概率，精修归 TRM）。
双时间尺度：h^L 每步快更新、h^H 每 K 步慢更新，层级特征 → multi-label softmax 分类。"""
import numpy as np
from sklearn.linear_model import LogisticRegression

class HrmRegime:
    LABELS = ["H-mode", "ELM", "locked", "tearing", "impurity", "ramp-down"]
    def __init__(self, d_feat, K=5):
        self.K = K
        self.hL = np.zeros(d_feat); self.hH = np.zeros(d_feat); self._k = 0
        self.clf = {}
    def update(self, feat):
        """快态每步 EMA 更新；慢态每 K 步由快态聚合更新。"""
        feat = np.asarray(feat, float)
        self.hL = 0.6 * self.hL + 0.4 * feat
        self._k += 1
        if self._k % self.K == 0:
            self.hH = 0.8 * self.hH + 0.2 * self.hL
        return np.concatenate([self.hL, self.hH])
    def fit(self, feats, labels):
        """labels: {label: 0/1 序列}，每标签一个 logistic（multi-label）。
        C=100 弱正则：hL 量程~0.05，默认 C=1 的 L2 会把系数压成弱概率（不修量纲不放大噪声）。"""
        for lab in self.LABELS:
            y = np.asarray(labels[lab])
            if y.sum() in (0, len(y)):
                self.clf[lab] = ("const", float(y[0]))
            else:
                c = LogisticRegression(max_iter=1000, C=100.0, class_weight="balanced").fit(feats, y)
                self.clf[lab] = ("model", c)
        return self
    def infer_proba(self, hfeat):
        out = {}
        for lab in self.LABELS:
            kind, m = self.clf.get(lab, ("const", 0.0))
            out[lab] = m if kind == "const" else float(m.predict_proba(np.asarray(hfeat, float).reshape(1, -1))[0, 1])
        return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h = HrmRegime(3, K=3)
    feats, labels = [], {k: [] for k in HrmRegime.LABELS}
    for i in range(400):
        elm = i % 50 > 40
        x = np.array([0.1 * rng.standard_normal() + (0.12 if elm else 0.005),
                      0.3 + 0.4 * elm + 0.05 * rng.standard_normal(),
                      0.02 * rng.standard_normal()])
        feats.append(h.update(x))
        for k in HrmRegime.LABELS:
            labels[k].append(1 if (k == "ELM" and elm) or (k == "H-mode" and not elm) else 0)
    h.fit(np.array(feats)[5:], {k: v[5:] for k, v in labels.items()})
    for _ in range(25):   # EMA 沉淀（在线连续调用时自然满足）
        fe_elm = h.update(np.array([0.12, 0.7, 0.0]))
    for _ in range(25):
        fe_norm = h.update(np.array([0.005, 0.3, 0.0]))
    p_elm = h.infer_proba(fe_elm); p_norm = h.infer_proba(fe_norm)
    assert p_elm["ELM"] > 0.8 and p_norm["ELM"] < 0.2 and p_norm["H-mode"] > 0.8
    print("hrm OK:", {k: round(v, 2) for k, v in p_elm.items()})
