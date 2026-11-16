"""实时数据织物（Layer A）：多速率摄入 + provenance + 缺失不静默插值。"""
import time
from collections import deque
from ..schema import now_prov, Quantity
from .. import constants

class DataFabric:
    def __init__(self, maxlen=2000):
        self.buf = {}          # channel -> deque[Quantity]
        self.maxlen = maxlen
    def ingest(self, channel, value, t=None, quality=1.0, uncertainty=0.0):
        q = Quantity(value, now_prov(channel, uncertainty, quality, t=t))
        self.buf.setdefault(channel, deque(maxlen=self.maxlen)).append(q)
        return q
    def latest(self, channel):
        """最新量；缺失/超时 → imputed 标记 + 降信任（不静默）。"""
        dq = self.buf.get(channel)
        t_now = time.time()
        if not dq:
            return None
        q = dq[-1]
        amax = constants.get("age_max", {}).get(channel, 0.05)
        if q.prov.age(t_now) > amax:
            return q.with_value(q.value, imputation=True, uncertainty=max(q.prov.uncertainty, 0.15))
        return q
    def stale_channels(self):
        t_now = time.time()
        return [c for c, dq in self.buf.items()
                if dq[-1].prov.age(t_now) > constants.get("age_max", {}).get(c, 0.05)]

if __name__ == "__main__":
    f = DataFabric()
    f.ingest("spec", [1, 2, 3], t=time.time() - 10)   # 10s 前的光谱 → stale
    q = f.latest("spec")
    assert q.prov.imputation and q.prov.uncertainty >= 0.15
    assert "spec" in f.stale_channels()
    print("fabric OK")
