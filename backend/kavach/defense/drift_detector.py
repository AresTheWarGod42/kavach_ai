from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from river import drift, linear_model, tree


@dataclass
class DriftUpdate:
    drift_detected: bool
    reason: str


@dataclass
class OnlineDriftManager:
    adwin: drift.ADWIN = field(default_factory=drift.ADWIN)
    hoeffding: tree.HoeffdingTreeClassifier = field(default_factory=tree.HoeffdingTreeClassifier)
    online_lr: linear_model.LogisticRegression = field(default_factory=linear_model.LogisticRegression)
    fraud_rate_window: deque[int] = field(default_factory=lambda: deque(maxlen=5000))
    rolling_true: deque[int] = field(default_factory=lambda: deque(maxlen=5000))
    rolling_pred: deque[int] = field(default_factory=lambda: deque(maxlen=5000))
    drift_state: str = "stable"

    def update_unlabeled_fraud_rate(self, score: float) -> DriftUpdate:
        is_flagged = 1 if score >= 0.6 else 0
        self.fraud_rate_window.append(is_flagged)
        self.adwin.update(float(is_flagged))
        if self.adwin.drift_detected:
            self.drift_state = "drift_detected"
            return DriftUpdate(True, "ADWIN fraud-rate drift")
        if self.drift_state != "retrain_urgent":
            self.drift_state = "stable"
        return DriftUpdate(False, "no drift")

    def update_labeled(self, features: dict[str, float], y_true: int, score: float) -> None:
        y_pred = 1 if score >= 0.6 else 0
        self.rolling_true.append(int(y_true))
        self.rolling_pred.append(int(y_pred))
        self.hoeffding.learn_one(features, y_true)
        self.online_lr.learn_one(features, y_true)
        if len(self.rolling_true) > 100:
            recall = self._recall()
            if recall < 0.80:
                self.drift_state = "retrain_urgent"

    def _recall(self) -> float:
        tp = sum(1 for yt, yp in zip(self.rolling_true, self.rolling_pred) if yt == 1 and yp == 1)
        fn = sum(1 for yt, yp in zip(self.rolling_true, self.rolling_pred) if yt == 1 and yp == 0)
        return float(tp / max(tp + fn, 1))

    def metrics(self) -> dict[str, Any]:
        tp = sum(1 for yt, yp in zip(self.rolling_true, self.rolling_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(self.rolling_true, self.rolling_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(self.rolling_true, self.rolling_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(self.rolling_true, self.rolling_pred) if yt == 1 and yp == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        beta = 2.0
        f2 = (1 + beta * beta) * precision * recall / max((beta * beta * precision + recall), 1e-9)
        fpr = fp / max(fp + tn, 1)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f2": float(f2),
            "fpr": float(fpr),
            "drift_status": self.drift_state,
        }

