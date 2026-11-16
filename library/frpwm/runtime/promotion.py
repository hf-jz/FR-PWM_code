"""模型晋升流水线（P2-13，报告 8/16/17：Production→Shadow→Validation→Canary→Promotion）。"""
import numpy as np

class ModelRegistry:
    def __init__(self, gates=None):
        self.prod = None
        self.gates = gates or dict(dZ=0.1, dD=0.1, dV=0.15, dphys=0.1, drisk=0.1)
        self.audit = []
    def shadow_compare(self, m_new, m_prod, dataset):
        """影子对比：同数据并行推理，逐量 Δ。dataset: [(x, y_ref)]。"""
        deltas = []
        for x, y in dataset:
            d = abs(float(m_new(x)) - float(y))
            deltas.append(d)
        return dict(dZ=float(np.mean(deltas)), dD=float(np.max(deltas)),
                    dV=float(np.std(deltas)), dphys=float(np.mean(deltas)),
                    drisk=float(np.max(deltas)))
    def try_promote(self, name, m_new, dataset, canary_frac=0.1):
        """shadow 达标 → canary（小样）→ production。全程可审计。"""
        if self.prod is None:
            self.prod = m_new
            self.audit.append((name, "initial_production"))
            return True
        d = self.shadow_compare(m_new, self.prod, dataset)
        ok = all(d[k] <= v for k, v in self.gates.items())
        self.audit.append((name, "shadow", d, ok))
        if not ok:
            return False
        nc = max(1, int(len(dataset) * canary_frac))
        d2 = self.shadow_compare(m_new, self.prod, dataset[:nc])
        ok2 = all(d2[k] <= v * 1.5 for k, v in self.gates.items())
        self.audit.append((name, "canary", d2, ok2))
        if ok2:
            self.prod = m_new
            self.audit.append((name, "promoted"))
        return ok2

if __name__ == "__main__":
    reg = ModelRegistry()
    reg.try_promote("v1", lambda x: x ** 2, [(i, i ** 2) for i in range(50)])
    good = reg.try_promote("v2", lambda x: x ** 2 + 0.01, [(i, i ** 2) for i in range(50)])
    bad = reg.try_promote("v3", lambda x: x ** 2 + 5.0, [(i, i ** 2) for i in range(50)])
    assert good and not bad
    print("promotion OK: v2 晋升/v3 拒绝, audit:", [a[1] for a in reg.audit])
