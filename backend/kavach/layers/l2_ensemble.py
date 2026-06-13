from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import shap

from kavach.config import settings


def _safe_sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def _conservative_score(score: float) -> float:
    gamma = max(1.0, float(settings.score_tolerance_gamma))
    return float(np.clip(np.power(np.clip(score, 0.0, 1.0), gamma), 0.0, 1.0))


@dataclass
class ONNXModel:
    path: Path
    session: ort.InferenceSession | None = None
    input_name: str | None = None
    output_name: str | None = None

    def load(self) -> None:
        if not self.path.exists():
            return
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, arr: np.ndarray) -> np.ndarray:
        if not self.session or not self.input_name or not self.output_name:
            return np.zeros((arr.shape[0],), dtype=np.float32)
        out = self.session.run([self.output_name], {self.input_name: arr.astype(np.float32)})[0]
        if out.ndim == 2 and out.shape[1] > 1:
            return out[:, 1].astype(np.float32)
        return out.reshape(-1).astype(np.float32)


@dataclass
class L2Ensemble:
    models_dir: Path
    xgb: ONNXModel = field(init=False)
    lgbm: ONNXModel = field(init=False)
    iso: ONNXModel = field(init=False)
    meta: ONNXModel = field(init=False)
    l2_features: list[str] = field(default_factory=list)
    request_center: np.ndarray | None = None
    request_scale: np.ndarray | None = None
    meta_threshold: float = 0.60
    platt: Any = None
    isotonic: Any = None
    calibration_min: float = 0.0
    calibration_max: float = 1.0
    shap_baseline_top5: list[str] = field(default_factory=list)
    shap_model: Any = None
    shap_explainer: Any = None

    def __post_init__(self) -> None:
        self.xgb = ONNXModel(self.models_dir / "l2_xgb.onnx")
        self.lgbm = ONNXModel(self.models_dir / "l2_lgbm.onnx")
        self.iso = ONNXModel(self.models_dir / "l2_iso.onnx")
        self.meta = ONNXModel(self.models_dir / "l2_meta.onnx")

    def load(self) -> None:
        self.xgb.load()
        self.lgbm.load()
        self.iso.load()
        self.meta.load()
        metadata_path = self.models_dir / "l2_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.l2_features = metadata.get("l2_features", [])
            self.meta_threshold = float(metadata.get("meta_threshold", self.meta_threshold))
            self.calibration_min = float(metadata.get("calibration_min", 0.0))
            self.calibration_max = float(metadata.get("calibration_max", 1.0))
            self.shap_baseline_top5 = metadata.get("shap_baseline_top5", [])
            center = metadata.get("request_center")
            scale = metadata.get("request_scale")
            if center and scale:
                self.request_center = np.asarray(center, dtype=np.float32)
                self.request_scale = np.asarray(scale, dtype=np.float32)
        xgb_pkl = self.models_dir / "l2_xgb.pkl"
        if xgb_pkl.exists():
            with open(xgb_pkl, "rb") as f:
                self.shap_model = pickle.load(f)
            try:
                self.shap_explainer = shap.TreeExplainer(self.shap_model)
            except Exception:
                self.shap_explainer = None

        platt_path = self.models_dir / "platt.pkl"
        if platt_path.exists():
            with open(platt_path, "rb") as f:
                self.platt = pickle.load(f)
        iso_path = self.models_dir / "isotonic.pkl"
        if iso_path.exists():
            with open(iso_path, "rb") as f:
                self.isotonic = pickle.load(f)

    def _vectorize(self, txn: dict[str, Any], features: dict[str, float]) -> tuple[np.ndarray, list[str]]:
        raw = {
            "amount": float(txn.get("amount", 0.0)),
            "txn_count_1m": float(features.get("txn_count_1m", 0.0)),
            "txn_count_5m": float(features.get("txn_count_5m", 0.0)),
            "txn_count_1h": float(features.get("txn_count_1h", 0.0)),
            "amount_sum_1h": float(features.get("amount_sum_1h", 0.0)),
            "unique_merchant_1h": float(features.get("unique_merchant_1h", 0.0)),
            "amount_zscore_30d": float(features.get("amount_zscore_30d", 0.0)),
            "hour_percentile_user": float(features.get("hour_percentile_user", 0.0)),
            "is_new_device": float(features.get("is_new_device", 0.0)),
            "device_txn_count": float(features.get("device_txn_count", 0.0)),
            "geographic_disparity": float(features.get("geographic_disparity", 0.0)),
            "unusual_location_flag": float(features.get("unusual_location_flag", 0.0)),
            "geographic_location_vs_ip": float(features.get("geographic_location_vs_ip", 0.0)),
            "merchant_category_risk_tier": float(features.get("merchant_category_risk_tier", 3.0)),
            "cross_category_anomaly": float(features.get("cross_category_anomaly", 0.0)),
            "hour_sin": float(features.get("hour_sin", 0.0)),
            "hour_cos": float(features.get("hour_cos", 1.0)),
            "otp_request_frequency": float(features.get("otp_request_frequency", 0.0)),
            "transaction_velocity": float(features.get("transaction_velocity", 0.0)),
            "failed_transaction_count": float(features.get("failed_transaction_count", 0.0)),
        }
        names = self.l2_features or list(raw.keys())
        arr = np.asarray([[raw.get(name, 0.0) for name in names]], dtype=np.float32)
        return arr, names

    def detect_request_outlier(self, feature_array: np.ndarray) -> bool:
        if self.request_center is None or self.request_scale is None:
            return False
        distance = np.abs((feature_array[0] - self.request_center) / np.maximum(self.request_scale, 1e-3))
        return bool(np.mean(distance) > 7.5 or np.max(distance) > 12.0)

    def _heuristic_base(self, arr: np.ndarray) -> tuple[float, float, float]:
        x = arr[0]
        amount = x[0]
        txn_1m = x[1]
        z = x[6]
        location = x[11]
        xgb = min(1.0, _safe_sigmoid(-2.6 + 0.000013 * amount + 0.22 * txn_1m + 0.45 * z + 0.35 * location))
        lgb = min(1.0, _safe_sigmoid(-2.9 + 0.000012 * amount + 0.28 * x[4] + 0.40 * x[14]))
        iso = min(1.0, max(0.0, 0.12 + 0.08 * txn_1m + 0.12 * abs(z)))
        return float(xgb), float(lgb), float(iso)

    def _calibrate(self, score: float) -> tuple[float, bool]:
        calibrated = score
        if self.platt is not None:
            calibrated = float(self.platt.predict_proba(np.asarray([[score]], dtype=np.float32))[0, 1])
        if self.isotonic is not None:
            calibrated = float(self.isotonic.predict(np.asarray([calibrated], dtype=np.float32))[0])
        audit_flag = bool(calibrated < self.calibration_min or calibrated > self.calibration_max)
        return min(1.0, max(0.0, calibrated)), audit_flag

    def _shap_like(self, names: list[str], arr: np.ndarray, score: float) -> list[dict[str, Any]]:
        if self.shap_explainer is not None:
            try:
                vals = self.shap_explainer.shap_values(arr)
                shap_arr = np.asarray(vals)
                if shap_arr.ndim == 3:
                    shap_arr = shap_arr[1]
                contrib = shap_arr[0]
                ranked_idx = np.argsort(np.abs(contrib))[::-1][:5]
                return [
                    {
                        "feature": names[i],
                        "feature_value": float(arr[0][i]),
                        "shap_value": float(contrib[i]),
                    }
                    for i in ranked_idx
                ]
            except Exception:
                pass
        values = arr[0]
        refs = np.asarray(self.request_center if self.request_center is not None else np.zeros_like(values), dtype=np.float32)
        scale = np.asarray(self.request_scale if self.request_scale is not None else np.ones_like(values), dtype=np.float32)
        contribution = (values - refs) / np.maximum(scale, 1e-3)
        ranked_idx = np.argsort(np.abs(contribution))[::-1][:5]
        out = []
        for idx in ranked_idx:
            out.append(
                {
                    "feature": names[idx],
                    "feature_value": float(values[idx]),
                    "shap_value": float(contribution[idx] * 0.08 * max(score, 0.1)),
                }
            )
        return out

    def score(self, txn: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
        arr, names = self._vectorize(txn, features)
        request_outlier_flag = self.detect_request_outlier(arr)

        xgb_score = float(self.xgb.predict(arr)[0]) if self.xgb.session else None
        lgb_score = float(self.lgbm.predict(arr)[0]) if self.lgbm.session else None
        iso_score = float(self.iso.predict(arr)[0]) if self.iso.session else None
        if xgb_score is None or lgb_score is None or iso_score is None:
            xgb_score, lgb_score, iso_score = self._heuristic_base(arr)

        meta_features = np.asarray(
            [
                [
                    xgb_score,
                    lgb_score,
                    iso_score,
                    float(features.get("txn_count_1m", 0.0)),
                    float(features.get("txn_count_5m", 0.0)),
                    float(features.get("txn_count_1h", 0.0)),
                    float(features.get("amount_sum_1h", 0.0)),
                ]
            ],
            dtype=np.float32,
        )
        if self.meta.session:
            raw_score = float(self.meta.predict(meta_features)[0])
        else:
            raw_score = float(np.clip(0.4 * xgb_score + 0.35 * lgb_score + 0.25 * iso_score, 0.0, 1.0))

        calibrated_score, calibration_audit_flag = self._calibrate(raw_score)
        tolerant_score = _conservative_score(calibrated_score)
        margin = max(0.03, 0.12 * (1.0 - abs(tolerant_score - 0.5)))
        ci = [max(0.0, tolerant_score - margin), min(1.0, tolerant_score + margin)]
        shap = self._shap_like(names, arr, tolerant_score) if tolerant_score >= settings.score_threshold_medium else []
        return {
            "score": tolerant_score,
            "confidence_interval": ci,
            "shap": shap,
            "base_model_scores": {"xgb": xgb_score, "lgbm": lgb_score, "isolation_forest": iso_score},
            "calibration_audit_flag": calibration_audit_flag,
            "request_outlier_flag": request_outlier_flag,
        }

    def shap_stability_check(self, candidate_top5: list[str]) -> bool:
        if not self.shap_baseline_top5:
            return True
        overlap = len(set(self.shap_baseline_top5).intersection(candidate_top5))
        return overlap >= 2
