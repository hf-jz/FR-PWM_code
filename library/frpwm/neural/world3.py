"""三段式世界模型（P1-10，报告 16 章：Short/Medium/Long + Projection+Calibration）。
Short(1-20 步): 神经算子混合模型；Medium(20-200): 物理锚定混合；Long(200+): 降阶物理。
段间交接做 Projection（状态映射）+ Calibration（首步残差标定），E_H^max<ε_H 分段验收。"""
import numpy as np
from .world import DualWorldModel, PhysicsStep
from .. import constants

class ThreeStageWorld:
    def __init__(self, n):
        self.short = DualWorldModel(n)          # 神经混合（短）
        self.medium = DualWorldModel(n)         # 混合（中）
        self.long_P = PhysicsStep(n, diff=0.02, decay=0.02, heat_gain=200.0)  # 降阶物理（长）
        self.n = n
        self._cal_m, self._cal_l = 1.0, 1.0
    def train(self, traj, epochs=120):
        self.short.train_residual(traj, epochs=epochs)
        self.medium.train_residual(traj, epochs=epochs)
        return self
    def _project_calibrate(self, X, y_teacher_next, stage):
        """Projection：段状态同空间直接映射；Calibration：用教师首步残差修正段增益。"""
        err = float(np.linalg.norm(X - y_teacher_next) / (np.linalg.norm(y_teacher_next) + 1e-30))
        if stage == "m":
            self._cal_m = float(np.clip(1.0 - err, 0.5, 1.5))
        else:
            self._cal_l = float(np.clip(1.0 - err, 0.5, 1.5))
        return X
    def rollout(self, X0, u, H, teacher=None):
        """分段 rollout：H≤20 短段；20<H≤200 中段；H>200 长段（含交接标定）。"""
        X = X0.copy(); preds = []
        for k in range(H):
            if k < 20:
                _, X = self.short.step(X, u)
            elif k < 200:
                if k == 20 and teacher is not None:
                    X = self._project_calibrate(X, teacher(X, 1), "m")
                _, X = self.medium.step(X, u)
                X = X * self._cal_m + X0 * (1 - self._cal_m)
            else:
                if k == 200 and teacher is not None:
                    X = self._project_calibrate(X, teacher(X, 1), "l")
                X = self.long_P(X, u) * self._cal_l + X * (1 - self._cal_l)
            preds.append(X.copy())
        return preds
    def segment_errors(self, X0, u, X_true_H, H):
        preds = self.rollout(X0, u, H)
        errs = [float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-30))
                for p, t in zip(preds, X_true_H)]
        segs = {"short": errs[:20], "medium": errs[20:200], "long": errs[200:]}
        return {k: (float(np.mean(v)) if v else 0.0, float(np.max(v)) if v else 0.0)
                for k, v in segs.items()}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    FP = PhysicsStep(n)
    true_step = lambda X, u: FP(X, u) + 30.0 * np.exp(-((np.linspace(0, 1, n) - 0.15) / 0.1) ** 2) * u
    X = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    traj = []
    for _ in range(400):
        u = rng.uniform(-1, 1); Xn = true_step(X, u)
        traj.append((X.copy(), u, Xn.copy())); X = Xn
    w3 = ThreeStageWorld(n).train(traj, epochs=80)
    X = 3000 * (1 - np.linspace(0, 1, n) ** 2) + 50
    Xs = []
    for _ in range(50):
        X = true_step(X, 0.5); Xs.append(X.copy())
    res = w3.segment_errors(traj[0][0], 0.5, Xs, 50)
    eps = constants.get("eps_H")
    assert res["short"][1] < eps.get(20, 0.10) and res["medium"][1] < 0.15, res
    print(f"world3 OK: short E^max={res['short'][1]:.3f} medium E^max={res['medium'][1]:.3f}")
