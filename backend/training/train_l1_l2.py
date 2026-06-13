from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from kavach.config import settings

try:
    from onnxmltools import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
except Exception:
    convert_lightgbm = None
    convert_xgboost = None
    FloatTensorType = None

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
except Exception:
    convert_sklearn = None
    SkFloatTensorType = None


TARGET_COL = "fraud_flag"
TIME_COL = "timestamp"


def normalize_dataset1_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "transaction id": "transaction_id",
        "transaction type": "transaction_type",
        "amount (INR)": "amount",
    }
    out = df.rename(columns=colmap).copy()
    out.columns = [c.strip().replace(" ", "_") for c in out.columns]
    return out


def prepare_dataset1_splits_and_exports(dataset_path: Path, export_path: Path) -> dict[str, Any]:
    df = normalize_dataset1_columns(pd.read_csv(dataset_path))
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).copy()
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df[TIME_COL].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df[TIME_COL].dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.drop_duplicates(subset=["transaction_id"])
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN")

    cap = df["amount"].quantile(0.999)
    df["amount"] = np.clip(df["amount"], 0, cap)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    n = len(df)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    train_df = df.iloc[:i1].copy()
    val_df = df.iloc[i1:i2].copy()
    test_df = df.iloc[i2:].copy()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(export_path, index=False)

    stats = {
        "train": train_df[TARGET_COL].value_counts(normalize=True).to_dict(),
        "val": val_df[TARGET_COL].value_counts(normalize=True).to_dict(),
        "test": test_df[TARGET_COL].value_counts(normalize=True).to_dict(),
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    return {"train": train_df, "val": val_df, "test": test_df, "stats": stats}


def _encode_categoricals(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cat_cols = [
        "transaction_type",
        "merchant_category",
        "transaction_status",
        "sender_age_group",
        "receiver_age_group",
        "sender_state",
        "sender_bank",
        "receiver_bank",
        "device_type",
        "network_type",
    ]
    enc_meta: dict[str, Any] = {}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        cardinality = train_df[col].nunique()
        if cardinality > 20:
            means = train_df.groupby(col)[TARGET_COL].mean().to_dict()
            global_mean = float(train_df[TARGET_COL].mean())
            train_df[col] = train_df[col].map(means).fillna(global_mean)
            val_df[col] = val_df[col].map(means).fillna(global_mean)
            test_df[col] = test_df[col].map(means).fillna(global_mean)
            enc_meta[col] = {"type": "target", "mapping": means, "global": global_mean}
        else:
            classes = {v: i for i, v in enumerate(sorted(train_df[col].astype(str).unique().tolist()))}
            unk = len(classes)
            train_df[col] = train_df[col].astype(str).map(classes).fillna(unk)
            val_df[col] = val_df[col].astype(str).map(classes).fillna(unk)
            test_df[col] = test_df[col].astype(str).map(classes).fillna(unk)
            enc_meta[col] = {"type": "label", "mapping": classes, "unknown": unk}
    return train_df, val_df, test_df, enc_meta


def _feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    drop_cols = [TARGET_COL, "transaction_id", TIME_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y


def _f2_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f2 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        yp = (y_score >= t).astype(int)
        f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = float(f2)
            best_t = float(t)
    return best_t, best_f2


def _export_onnx(model: Any, path: Path, kind: str, n_features: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "xgb" and convert_xgboost and FloatTensorType:
        onnx_model = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))])
        path.write_bytes(onnx_model.SerializeToString())
        return
    if kind == "lgbm" and convert_lightgbm and FloatTensorType:
        onnx_model = convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, n_features]))])
        path.write_bytes(onnx_model.SerializeToString())
        return
    if convert_sklearn and SkFloatTensorType:
        onnx_model = convert_sklearn(model, initial_types=[("input", SkFloatTensorType([None, n_features]))])
        path.write_bytes(onnx_model.SerializeToString())
        return
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(model, f)


def train_and_export_l1_l2(dataset_path: Path, models_dir: Path, export_path: Path) -> dict[str, float]:
    split = prepare_dataset1_splits_and_exports(dataset_path, export_path)
    train_df, val_df, test_df = split["train"], split["val"], split["test"]
    train_df, val_df, test_df, enc_meta = _encode_categoricals(train_df, val_df, test_df)

    X_train_raw, y_train = _feature_target(train_df)
    X_val, y_val = _feature_target(val_df)
    X_test, y_test = _feature_target(test_df)

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train_raw, y_train)
    adasyn = ADASYN(random_state=42)
    X_ad, y_ad = adasyn.fit_resample(X_sm, y_sm)
    tomek = TomekLinks()
    X_res, y_res = tomek.fit_resample(X_ad, y_ad)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    l1_xgb = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    l1_xgb.fit(X_res, y_res)

    xgb = XGBClassifier(
        n_estimators=240,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=260,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(n_estimators=180, contamination=0.03, random_state=42)
    xgb.fit(X_res, y_res)
    lgbm.fit(X_res, y_res)
    iso.fit(X_res)

    xgb_val = xgb.predict_proba(X_val)[:, 1]
    lgb_val = lgbm.predict_proba(X_val)[:, 1]
    iso_val = -iso.score_samples(X_val)
    iso_val = (iso_val - iso_val.min()) / max((iso_val.max() - iso_val.min()), 1e-6)
    meta_features_val = np.c_[
        xgb_val,
        lgb_val,
        iso_val,
        X_val["txn_count_1m"].to_numpy() if "txn_count_1m" in X_val.columns else np.zeros(len(X_val)),
        X_val["txn_count_5m"].to_numpy() if "txn_count_5m" in X_val.columns else np.zeros(len(X_val)),
        X_val["txn_count_1h"].to_numpy() if "txn_count_1h" in X_val.columns else np.zeros(len(X_val)),
        X_val["amount_sum_1h"].to_numpy() if "amount_sum_1h" in X_val.columns else np.zeros(len(X_val)),
    ]
    meta = LogisticRegression(C=0.4, penalty="l2", max_iter=2000)
    meta.fit(meta_features_val, y_val)
    raw_val_scores = meta.predict_proba(meta_features_val)[:, 1]

    platt = LogisticRegression(max_iter=1000)
    platt.fit(raw_val_scores.reshape(-1, 1), y_val)
    platt_scores = platt.predict_proba(raw_val_scores.reshape(-1, 1))[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(platt_scores, y_val)
    calibrated_val = isotonic.predict(platt_scores)
    threshold, best_f2 = _f2_best_threshold(y_val, calibrated_val)

    xgb_test = xgb.predict_proba(X_test)[:, 1]
    lgb_test = lgbm.predict_proba(X_test)[:, 1]
    iso_test = -iso.score_samples(X_test)
    iso_test = (iso_test - iso_test.min()) / max((iso_test.max() - iso_test.min()), 1e-6)
    meta_features_test = np.c_[
        xgb_test,
        lgb_test,
        iso_test,
        X_test["txn_count_1m"].to_numpy() if "txn_count_1m" in X_test.columns else np.zeros(len(X_test)),
        X_test["txn_count_5m"].to_numpy() if "txn_count_5m" in X_test.columns else np.zeros(len(X_test)),
        X_test["txn_count_1h"].to_numpy() if "txn_count_1h" in X_test.columns else np.zeros(len(X_test)),
        X_test["amount_sum_1h"].to_numpy() if "amount_sum_1h" in X_test.columns else np.zeros(len(X_test)),
    ]
    raw_test_scores = meta.predict_proba(meta_features_test)[:, 1]
    calibrated_test = isotonic.predict(platt.predict_proba(raw_test_scores.reshape(-1, 1))[:, 1])
    y_pred = (calibrated_test >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    auc_pr = average_precision_score(y_test, calibrated_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)

    # FGSM-style tabular adversarial hardening with epsilon=0.01 using a differentiable surrogate.
    fgsm_probe = LogisticRegression(max_iter=1200)
    X_res_np = X_res.to_numpy(dtype=np.float32)
    fgsm_probe.fit(X_res_np, y_res)
    probe_proba = fgsm_probe.predict_proba(X_res_np)[:, 1]
    grad_direction = np.sign((probe_proba - y_res).reshape(-1, 1) * fgsm_probe.coef_)
    X_adv = np.clip(
        X_res_np + 0.01 * grad_direction,
        a_min=np.min(X_res_np, axis=0),
        a_max=np.max(X_res_np, axis=0),
    )
    X_adv_df = pd.DataFrame(X_adv, columns=X_res.columns)
    xgb.fit(pd.concat([X_res, X_adv_df], ignore_index=True), np.concatenate([y_res, y_res]))
    lgbm.fit(pd.concat([X_res, X_adv_df], ignore_index=True), np.concatenate([y_res, y_res]))

    # SHAP baseline top5 for stability checks.
    explainer = shap.TreeExplainer(xgb)
    sample = X_val.sample(n=min(300, len(X_val)), random_state=42)
    shap_values = explainer.shap_values(sample)
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[1]
    mean_importance = np.abs(shap_arr).mean(axis=0)
    top_idx = np.argsort(mean_importance)[::-1][:5]
    top5_features = [sample.columns[i] for i in top_idx]

    # Adversarial evasive suite (50 crafted transactions).
    evasive = X_test.sample(n=min(50, len(X_test)), random_state=7).copy()
    for col in ["txn_count_1m", "txn_count_5m", "txn_count_1h"]:
        if col in evasive.columns:
            evasive[col] = np.minimum(evasive[col], 1.0)
    if "is_new_device" in evasive.columns:
        evasive["is_new_device"] = 0.0
    if "amount_zscore_30d" in evasive.columns:
        evasive["amount_zscore_30d"] = np.clip(evasive["amount_zscore_30d"], -0.5, 1.2)
    xe = xgb.predict_proba(evasive)[:, 1]
    le = lgbm.predict_proba(evasive)[:, 1]
    ie = -iso.score_samples(evasive)
    ie = (ie - ie.min()) / max(ie.max() - ie.min(), 1e-6)
    me = np.c_[xe, le, ie, np.zeros((len(evasive), 4))]
    evasive_score = isotonic.predict(platt.predict_proba(meta.predict_proba(me)[:, 1].reshape(-1, 1))[:, 1])
    evasive_catch_rate = float((evasive_score >= threshold).mean())

    models_dir.mkdir(parents=True, exist_ok=True)
    _export_onnx(l1_xgb, models_dir / "l1_xgb.onnx", "xgb", X_res.shape[1])
    _export_onnx(xgb, models_dir / "l2_xgb.onnx", "xgb", X_res.shape[1])
    _export_onnx(lgbm, models_dir / "l2_lgbm.onnx", "lgbm", X_res.shape[1])
    _export_onnx(iso, models_dir / "l2_iso.onnx", "sklearn", X_res.shape[1])
    _export_onnx(meta, models_dir / "l2_meta.onnx", "sklearn", meta_features_val.shape[1])
    with open(models_dir / "l2_xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(models_dir / "l2_lgbm.pkl", "wb") as f:
        pickle.dump(lgbm, f)

    with open(models_dir / "platt.pkl", "wb") as f:
        pickle.dump(platt, f)
    with open(models_dir / "isotonic.pkl", "wb") as f:
        pickle.dump(isotonic, f)
    with open(models_dir / "encoders_dataset1.pkl", "wb") as f:
        pickle.dump(enc_meta, f)

    metadata = {
        "l2_features": X_res.columns.tolist(),
        "meta_threshold": threshold,
        "calibration_min": float(np.min(calibrated_val)),
        "calibration_max": float(np.max(calibrated_val)),
        "request_center": X_train_raw.mean().to_list(),
        "request_scale": (X_train_raw.std().replace(0, 1.0)).to_list(),
        "shap_baseline_top5": top5_features,
        "class_weight_scale_pos": scale_pos_weight,
        "smote_adasyn_tomek": {"smote": len(X_sm), "adasyn": len(X_ad), "tomek": len(X_res)},
    }
    (models_dir / "l2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    prev_baseline_path = models_dir / "shap_baseline_prev.json"
    if prev_baseline_path.exists():
        prev_top = json.loads(prev_baseline_path.read_text(encoding="utf-8")).get("top5", [])
        overlap = len(set(prev_top).intersection(top5_features))
        review_flag = overlap < 2
    else:
        review_flag = False
    (models_dir / "shap_baseline_prev.json").write_text(json.dumps({"top5": top5_features}, indent=2), encoding="utf-8")

    provenance_rows = [
        {
            "source_dataset": str(dataset_path),
            "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
            "label_origin": TARGET_COL,
            "synthetic_flag": 0,
            "synthetic_method": "",
            "model_version": settings.model_version,
        },
        {
            "source_dataset": str(dataset_path),
            "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
            "label_origin": TARGET_COL,
            "synthetic_flag": 1,
            "synthetic_method": "SMOTE+ADASYN",
            "model_version": settings.model_version,
        },
    ]
    pd.DataFrame(provenance_rows).to_csv(models_dir / "provenance_dataset1.csv", index=False)

    # Stratified K-Fold CV with in-fold balancing pipeline.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f2 = []
    for tr_idx, va_idx in cv.split(X_train_raw, y_train):
        X_tr, X_va = X_train_raw.iloc[tr_idx], X_train_raw.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        X_sm_f, y_sm_f = smote.fit_resample(X_tr, y_tr)
        X_ad_f, y_ad_f = adasyn.fit_resample(X_sm_f, y_sm_f)
        X_to_f, y_to_f = tomek.fit_resample(X_ad_f, y_ad_f)
        model_cv = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        model_cv.fit(X_to_f, y_to_f)
        pred = (model_cv.predict_proba(X_va)[:, 1] >= 0.2).astype(int)
        cv_f2.append(float(fbeta_score(y_va, pred, beta=2, zero_division=0)))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("Kavach.ai-L1-L2")
    with mlflow.start_run(run_name=f"{settings.model_version}-dataset1"):
        mlflow.log_params(
            {
                "scale_pos_weight": scale_pos_weight,
                "threshold_f2": threshold,
                "cv_f2_mean": float(np.mean(cv_f2)),
                "fgsm_epsilon": 0.01,
                "calibration": "platt+isotonic",
            }
        )
        mlflow.log_metrics(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f2": float(f2),
                "auc_pr": float(auc_pr),
                "fpr": float(fpr),
                "adversarial_catch_rate": evasive_catch_rate,
                "shap_review_alert": float(review_flag),
            }
        )
        mlflow.log_artifact(str(export_path))
        mlflow.log_artifact(str(models_dir / "l2_metadata.json"))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "auc_pr": float(auc_pr),
        "fpr": float(fpr),
        "threshold": float(threshold),
        "adversarial_catch_rate": evasive_catch_rate,
    }


def main() -> None:
    split = prepare_dataset1_splits_and_exports(settings.dataset1_path, settings.dataset1_export_path)
    print("[Dataset1] train distribution:", split["stats"]["train"])
    print("[Dataset1] val distribution:", split["stats"]["val"])
    print("[Dataset1] test distribution:", split["stats"]["test"])
    metrics = train_and_export_l1_l2(settings.dataset1_path, settings.models_dir, settings.dataset1_export_path)
    print("[Dataset1] metrics:", metrics)


if __name__ == "__main__":
    main()
