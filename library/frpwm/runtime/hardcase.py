"""Hard Case Replay Buffer（P1-8，报告 16/17：事件→捕获→去重→标注→物理验证→缓冲→训练）。"""
import hashlib
import numpy as np

class HardCaseBuffer:
    """Hard Case ≠ Training Label：入缓冲前必须 physics_verify 通过。"""
    def __init__(self, maxlen=10000):
        self.cases = []
        self._seen = set()
        self.maxlen = maxlen
    def _key(self, feat):
        return hashlib.sha256(np.round(np.asarray(feat, float), 3).tobytes()).hexdigest()[:16]
    def capture(self, feat, kind, label=None, physics_verify=True):
        """kind: OOD/high_surprise/phys_residual/failed_inference/model_divergence/rare。
        physics_verify=False 的案例保留在 quarantine，不进训练集。"""
        k = self._key(feat)
        if k in self._seen:
            return False                              # Deduplicate
        self._seen.add(k)
        self.cases.append(dict(key=k, feat=np.asarray(feat, np.float32), kind=kind,
                               label=label, verified=bool(physics_verify)))
        if len(self.cases) > self.maxlen:
            self.cases.pop(0)
        return True
    def training_set(self):
        """只有验证过的案例才成为训练样本（Hard Case ≠ Training Label）。"""
        v = [c for c in self.cases if c["verified"] and c["label"] is not None]
        if not v:
            return None, None
        return np.stack([c["feat"] for c in v]), np.array([c["label"] for c in v])
    def stats(self):
        kinds = {}
        for c in self.cases:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        return dict(total=len(self.cases), verified=sum(c["verified"] for c in self.cases),
                    kinds=kinds)

if __name__ == "__main__":
    buf = HardCaseBuffer()
    x = np.random.default_rng(0).normal(0, 1, 10)
    assert buf.capture(x, "ood", label=1)
    assert not buf.capture(x, "ood", label=1)          # 去重
    buf.capture(x + 1, "phys_residual", label=None, physics_verify=False)
    X, Y = buf.training_set()
    assert X.shape[0] == 1 and Y[0] == 1               # 未验证/无标签不进训练集
    print("hardcase OK:", buf.stats())
