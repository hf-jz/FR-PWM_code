"""PredictionCache（2.1.3）：World Model 异步写快照，MPC 只读最近快照，绝不同步调用。
快照八元组 + age 门 + source revision 门（快照-状态链绑定）；超门降级 PhysicsOnlyPredictor。
PhysicsOnlyPredictor：Φ^P_Δt 纯物理积分（无 R_φ），Jacobian 离线预计算、在线 O(1) 查表。"""
import time
from dataclasses import dataclass, field
import numpy as np
from .. import constants

@dataclass
class PredictionSnapshot:
    X_hat: np.ndarray                    # 预测轨迹 [H, n]
    prediction_time: float
    available_time: float
    model_bundle: str
    sigma_wm: np.ndarray = None          # World Model 不确定度
    source_state_id: int = 0
    source_state_revision: int = 0
    source_bundle_id: str = ""
    @property
    def prediction_age(self):
        return time.time() - self.prediction_time

class PredictionCache:
    """单写者-多读者最新快照缓存；age 门与 revision 门同时通过才可消费。"""
    def __init__(self, age_max=None):
        self.age_max = constants.get("age_max", {}).get("spec", 0.020) if age_max is None else age_max
        self._snap = None
    def write(self, snap):
        self._snap = snap
    @staticmethod
    def revision_compatible(src_rev, cur_rev, bundle_ok=True):
        """source revision 是当前 revision 的祖先或相等（同一状态链），且 bundle 一致。"""
        return bundle_ok and 0 <= src_rev <= cur_rev
    def read(self, cur_state_revision, cur_bundle_id, t_now=None):
        """双硬门：age ≤ age_max 且 revision/bundle 兼容 → (snap, ok)；否则 (snap, False)。"""
        s = self._snap
        if s is None:
            return None, False
        age = (t_now or time.time()) - s.prediction_time
        if age > self.age_max:
            return s, False
        ok = self.revision_compatible(s.source_state_revision, cur_state_revision,
                                      bundle_ok=(s.source_bundle_id == cur_bundle_id))
        return s, bool(ok)

def nz_q_authoritative(fast, slow=None, cur_revision=0, cur_bundle_id="",
                       fast_revision=None, slow_revision=None, slow_bundle_id="",
                       regime_continuous=True, gates_ok=True):
    """n_z^q 双来源合并（2.2 冻结，报告 15 章/16 章模块 3）：缺省 authoritative = 单帧快通道值；
    慢环 refined revision 仅在满足快照-状态链绑定（source revision 为当前 revision 的祖先或相等、
    中间无 regime 跳变、bundle 一致）且过三硬门与 Physics-to-Observation closure 时才允许覆盖；
    不兼容即丢弃慢环版本——同一分量单值单来源，禁止两来源各自对外发布。
    → (value, revision, source ∈ {fast, slow_refined})
    """
    if fast_revision is None:
        fast_revision = cur_revision
    if slow is None:
        return fast, fast_revision, "fast"
    src_rev = cur_revision if slow_revision is None else slow_revision
    rev_ok = PredictionCache.revision_compatible(src_rev, cur_revision,
                                                 bundle_ok=(slow_bundle_id == cur_bundle_id))
    if rev_ok and regime_continuous and gates_ok:
        return slow, src_rev, "slow_refined"
    return fast, fast_revision, "fast"

class PhysicsOnlyPredictor:
    """physics-only 短视预测：Φ^P(X,U,M)=纯物理积分（不含 R_φ）。
    规格（2.1.3 冻结）：H_P≤5 步；不确定度 Σ_{k+1}=A_kΣA_k^T+Q_k；
    A_k=J_Φ 离线预计算（工作点网格，低秩/对角压缩），在线 O(1) 查表插值，禁在线有限差分。"""
    def __init__(self, physics_step, n, n_grid=16, q_proc=1.0):
        """physics_step: F_P(X,u) 可调用（如 neural.world.PhysicsStep）。
        init 时按工作点网格预计算压缩 Jacobian（对角近似 = 低秩压缩的 MVP 形式）。"""
        self.FP = physics_step
        self.n = n
        self.q = float(q_proc)
        rng = np.random.default_rng(0)
        base = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
        eps = 1e-3
        self._j_grid, self._j_diag = [], []
        for g in np.linspace(0.2, 1.8, n_grid):
            Xg = base * g
            f0 = np.asarray(physics_step(Xg, 0.0), float)
            # 逐维有限差分（离线 init 一次性，不在线调用）
            diag = np.empty(n)
            for i in range(n):
                dX = np.zeros(n); dX[i] = eps
                diag[i] = (np.asarray(physics_step(Xg + dX, 0.0), float)[i] - f0[i]) / eps
            self._j_grid.append(g); self._j_diag.append(diag)
        self._j_grid = np.array(self._j_grid); self._j_diag = np.array(self._j_diag)
    def jacobian_diag(self, X):
        """O(1) 查表插值：按 ‖X‖/‖base‖ 比例因子取最近工作点（压缩对角 J）。"""
        g = float(np.linalg.norm(X) / (np.linalg.norm(3000 * (1 - np.linspace(0, 1, self.n) ** 2) + 50) + 1e-30))
        i = int(np.argmin(np.abs(self._j_grid - g)))
        return self._j_diag[i]
    def predict(self, X, u=0.0, H=None, sigma0=None):
        """→ (轨迹 [H,n], Σ 轨迹 [H,n])。禁用条件由调用方判（F_P 输入缺失/ident=2 不得使用）。"""
        H = min(H or 5, 5)
        X = np.asarray(X, float)
        Jd = self.jacobian_diag(X)
        traj, sig = [], []
        S2 = np.full(self.n, float(sigma0 if sigma0 is not None else 1.0))
        for _ in range(H):
            X = np.asarray(self.FP(X, u), float)
            S2 = (Jd ** 2) * S2 + self.q          # Σ_{k+1}=AΣAᵀ+Q（对角压缩）
            traj.append(X.copy()); sig.append(np.sqrt(S2).copy())
        return np.array(traj), np.array(sig)

if __name__ == "__main__":
    from ..neural.world import PhysicsStep
    # PredictionCache 双门
    pc = PredictionCache(age_max=0.020)
    snap = PredictionSnapshot(X_hat=np.zeros((5, 3)), prediction_time=time.time(),
                              available_time=time.time(), model_bundle="b1",
                              source_state_id=105, source_state_revision=2, source_bundle_id="B")
    pc.write(snap)
    s, ok = pc.read(cur_state_revision=2, cur_bundle_id="B")
    assert ok and s is snap                                  # 祖先/相等 → 兼容
    _, ok2 = pc.read(cur_state_revision=1, cur_bundle_id="B")
    assert not ok2                                           # source 在未来之后 → 不兼容
    _, ok3 = pc.read(cur_state_revision=3, cur_bundle_id="C")
    assert not ok3                                           # bundle 变更 → 不兼容
    s4, ok4 = pc.read(cur_state_revision=3, cur_bundle_id="B")
    assert ok4                                               # 祖先 → 兼容
    import time as _t
    snap2 = PredictionSnapshot(X_hat=np.zeros((5, 3)), prediction_time=time.time() - 1.0,
                               available_time=time.time(), model_bundle="b1",
                               source_state_revision=3, source_bundle_id="B")
    pc.write(snap2)
    _, ok5 = pc.read(cur_state_revision=3, cur_bundle_id="B")
    assert not ok5                                           # age 超门
    # n_z^q 双来源合并门（2.2 冻结）
    fast = np.array([1.0, 2.0]); slow = np.array([1.1, 2.1])
    v, r, s = nz_q_authoritative(fast, cur_revision=5)
    assert s == "fast" and v is fast and r == 5              # 无慢环候选 → 快通道权威
    v, r, s = nz_q_authoritative(fast, slow=slow, cur_revision=5, cur_bundle_id="B",
                                 slow_revision=5, slow_bundle_id="B")
    assert s == "slow_refined" and v is slow and r == 5      # 祖先/相等 + 门全过 → 允许覆盖
    v, r, s = nz_q_authoritative(fast, slow=slow, cur_revision=5, cur_bundle_id="B",
                                 slow_revision=6, slow_bundle_id="B")
    assert s == "fast" and v is fast                         # 慢环 revision 在未来 → 丢弃
    v, r, s = nz_q_authoritative(fast, slow=slow, cur_revision=5, cur_bundle_id="B",
                                 slow_revision=5, slow_bundle_id="C")
    assert s == "fast"                                       # bundle 变更 → 丢弃
    v, r, s = nz_q_authoritative(fast, slow=slow, cur_revision=5, cur_bundle_id="B",
                                 slow_revision=5, slow_bundle_id="B", regime_continuous=False)
    assert s == "fast"                                       # regime 跳变 → 丢弃
    v, r, s = nz_q_authoritative(fast, slow=slow, cur_revision=5, cur_bundle_id="B",
                                 slow_revision=5, slow_bundle_id="B", gates_ok=False)
    assert s == "fast"                                       # 三硬门/闭环未过 → 丢弃
    # PhysicsOnlyPredictor
    n = 30
    pop = PhysicsOnlyPredictor(PhysicsStep(n), n)
    X0 = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    traj, sig = pop.predict(X0, u=0.5, H=5)
    assert traj.shape == (5, n) and sig.shape == (5, n)
    assert np.all(sig[-1] >= sig[0])                         # Σ 单调传播
    assert np.all(np.isfinite(traj))
    t0 = time.perf_counter()
    for _ in range(20):
        pop.predict(X0, u=0.5, H=5)
    ms = (time.perf_counter() - t0) / 20 * 1000
    assert ms < 1.0, ms                                      # ≤1ms 预算（CPU 确定性积分+查表）
    print(f"prediction_cache OK: 双门 5/5，PhysicsOnlyPredictor {ms:.3f}ms/5步 Σ 单调")
