"""集成不确定度 + 保形校准（P0-3，报告 17 模块 8：N 模型集成 + conformal 区间 + PICP₉₀）。
2.1.3 冻结增补：Conformal Safety Module regime×action 分层标定（n_min 回退全局）+ E_rel/E_σ 双指标门。"""
import numpy as np
from ..neural.world import DualWorldModel
from .. import constants

def _qstat(arr, level=0.90):
    """有限样本顺序统计量 ⌈(n+1)(1−ε)⌉（冻结定义，与 calibrate 同源）。"""
    a = np.sort(np.asarray(arr, float).ravel())
    k = int(np.ceil((a.size + 1) * level))
    return float(a[min(max(k, 1), a.size) - 1])

class StratifiedConformal:
    """regime×action region 分层标定（2.1.3 Conformal Safety Module 冻结）。
    每层独立 q̂；层样本 < conf_n_min（shot 级）时回退上一层级：全局 q̂ × conf_gamma_pool 保守膨胀。"""
    def __init__(self, level=0.90):
        self.level = level
        self.n_min = constants.get("conf_n_min", 200)
        self.gamma_pool = constants.get("conf_gamma_pool", 1.25)
        self.q_global = None
        self.q_layer = {}
    def fit(self, scores, keys):
        """scores: 残差数组；keys: 等长分层键（如 (regime, action_region) 元组）。"""
        scores = np.asarray(scores, float)
        keys = list(keys)
        assert len(scores) == len(keys)
        self.q_global = _qstat(scores, self.level)
        self.q_layer = {}
        for key in set(keys):
            s = scores[[k == key for k in keys]]
            self.q_layer[key] = (_qstat(s, self.level) if s.size >= self.n_min
                                 else self.q_global * self.gamma_pool)   # 分层不足 → 保守回退
        return self
    def q_hat(self, key):
        assert self.q_global is not None
        return float(self.q_layer.get(key, self.q_global * self.gamma_pool))

def safety_scores(g_true, g_pred, scale=None):
    """L2 conformal 非一致性分数（2.1.3 FVP 冻结：预测残差，严禁对 g(X) 本身求分位）。
    g_true/g_pred: (N, ...) 真值与预测约束函数值 → 单边残差 g(X)−g(X̂)（可选 scale 归一）。"""
    s = np.asarray(g_true, float) - np.asarray(g_pred, float)
    if scale is not None:
        s = s / np.asarray(scale, float)
    return s

def dual_gate(e_rel, e_sigma_by_domain, k=None):
    """代理误差双指标门（2.1.3 冻结）：E_rel<1% 且每一分域 E_σ < k（constants.e_sigma_k）。"""
    k = constants.get("e_sigma_k", 3.0) if k is None else k
    return bool(e_rel < 0.01 and all(s < k for s in dict(e_sigma_by_domain).values()))

class EnsembleWorld:
    def __init__(self, n, n_models=5):
        self.models = [DualWorldModel(n, seed=s) for s in range(n_models)]
        self.q90 = None
    def train(self, traj, epochs=120):
        """同一轨迹 bootstrap 训练 N 个模型（集成分歧 = epistemic）。"""
        rng = np.random.default_rng(0)
        for m in self.models:
            idx = rng.integers(0, len(traj), len(traj))
            m.train_residual([traj[i] for i in idx], epochs=epochs)
        return self
    def predict(self, X, u=0.0):
        """→ (μ, σ_epi)。σ 逐点集成分歧。"""
        preds = np.array([m.step(X, u)[1] for m in self.models])
        return preds.mean(0), preds.std(0)
    def calibrate(self, traj_cal, inflate=1.15):
        """保形校准：残差 |y−μ| 的 90% 分位 → 区间半宽 q90。
        有限样本指标冻结（2.1.3）：split-conformal 第 ⌈(n+1)(1−ε)⌉ 个顺序统计量。
        inflate：有限样本保守膨胀因子（校准/测试分布漂移的余量）。"""
        res = []
        for X, u, Y in traj_cal:
            mu, _ = self.predict(X, u)
            res.append(np.abs(Y - mu) / (np.abs(Y) + 1e-9))
        arr = np.sort(np.concatenate(res))
        k = int(np.ceil((arr.size + 1) * 0.90))          # ⌈(n+1)(1−ε)⌉ 顺序统计量
        self.q90 = float(arr[min(max(k, 1), arr.size) - 1]) * inflate
        return self.q90
    def picp(self, traj_test):
        """PICP₉₀：真值点落在 [μ(1−q90), μ(1+q90)] 内的逐点比例（非全剖面命中）。"""
        assert self.q90 is not None
        hit, n = [], 0
        for X, u, Y in traj_test:
            mu, _ = self.predict(X, u)
            rel = np.abs(Y - mu) / (np.abs(Y) + 1e-9)
            hit.append((rel <= self.q90).mean()); n += 1
        return float(np.mean(hit))
    def picp_clustered(self, shots):
        """shot 级聚类覆盖（2.1.3 统计单位冻结）：一炮内时间采样强相关，不作独立 Bernoulli——
        主验收单位 = shot 级（每炮的逐点命中率为一个聚类样本），附 block bootstrap 标准误。
        shots: {shot_id: traj}。→ (shot 命中率数组, 均值, bootstrap 95% 置信半径)。"""
        assert self.q90 is not None
        per_shot = {sid: self.picp(tr) for sid, tr in shots.items()}
        vals = np.array(list(per_shot.values()))
        rng = np.random.default_rng(0)
        boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(2000)]
        radius = float(np.quantile(boots, 0.975) - np.quantile(boots, 0.025)) / 2
        return vals, float(vals.mean()), radius
    def trajectory_score(self, X0, u_seq, Y_seq, scales):
        """trajectory-level conformal（2.1.3 算法冻结）：S_traj=max_k |e_k|/scale_k。
        X0 初始状态、u_seq/Y_seq 动作与真值序列、scales 逐步 conformal 尺度。"""
        X = X0.copy(); errs = []
        for k, (u, Y) in enumerate(zip(u_seq, Y_seq)):
            _, X = self.predict_step(X, u)
            errs.append(float(np.linalg.norm(Y - X) / (np.linalg.norm(Y) + 1e-30)) / scales[k])
        return float(np.max(errs))
    def predict_step(self, X, u):
        """单步便捷接口（集成均值 + 当前模型推进）。"""
        mu, s = self.predict(X, u)
        return mu, X + (mu - X)

if __name__ == "__main__":
    from ..neural.world import PhysicsStep
    rng = np.random.default_rng(0)
    n = 30
    FP = PhysicsStep(n)
    true_step = lambda X, u: FP(X, u) + 30.0 * np.exp(-((np.linspace(0, 1, n) - 0.15) / 0.1) ** 2) * u
    X = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    traj = []
    for _ in range(500):
        u = rng.uniform(-1, 1); Xn = true_step(X, u)
        traj.append((X.copy(), u, Xn.copy())); X = Xn
    ens = EnsembleWorld(n, 5).train(traj[:400], epochs=80)
    q90 = ens.calibrate(traj[400:450])
    cov = ens.picp(traj[450:])
    mu, s = ens.predict(traj[0][0], 0.5)
    assert 0.80 <= cov <= 1.0, cov
    assert s.mean() > 0
    # 2.1.3 shot 级聚类覆盖 + trajectory conformal
    shots = {f"shot{i}": traj[450 + i * 10: 460 + i * 10] for i in range(4)}
    vals, mean_c, radius = ens.picp_clustered(shots)
    assert len(vals) == 4 and 0.5 <= mean_c <= 1.0 and radius >= 0
    us = [0.5] * 5; Ys = [traj[460 + k][2] for k in range(5)]
    st = ens.trajectory_score(traj[460][0], us, Ys, scales=[max(q90, 1e-3)] * 5)
    assert st > 0
    print(f"ensemble OK: q90={q90:.4f} PICP={cov:.2f} σ_epi_mean={s.mean():.2f} "
          f"cluster={mean_c:.2f}±{radius:.3f} S_traj={st:.2f}")
