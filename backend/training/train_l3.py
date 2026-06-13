from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, fbeta_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from kavach.config import settings

try:
    import mlflow
except Exception:
    mlflow = None

try:
    from opacus import PrivacyEngine
except Exception:
    PrivacyEngine = None

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv

    HAS_PYG = True
except Exception:
    HAS_PYG = False


TARGET_COL = "is_fraud"


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 16) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def classify(self, emb):
        return self.head(emb).squeeze(-1)


KEEP_COLS = [
    "transaction_id",
    "user_id",
    "amount",
    "session_duration",
    "authentication_attempts",
    "receiver_account_age",
    "receiver_transaction_history",
    "transaction_amount_vs_sender_history",
    "geographic_disparity",
    "transaction_time_of_day",
    "merchant_category_code",
    "time_between_link_click_and_transaction",
    "unusual_device_flag",
    "unusual_ip_flag",
    "unusual_location_flag",
    "input_timing_consistency",
    "app_switching_frequency",
    "keyboard_input_speed",
    "input_pause_patterns",
    "screen_active_time",
    "geographic_location_vs_ip",
    "authentication_attempt_count",
    "time_between_otp_generation_and_input",
    "pin_entry_speed",
    "unusual_transaction_amount_flag",
    "otp_request_frequency",
    "transaction_velocity",
    "failed_transaction_count",
    "request_amount_roundness",
    "time_pressure_indicators",
    "handle_similarity_score",
    "handle_contains_official_terms",
    "handle_typo_analysis",
    "handle_verification_status",
    TARGET_COL,
]

LEAKAGE_FEATURE_ALIASES = {
    "d1",
    "d_1",
    "feature_d1",
    "time_between_otp_generation_and_input",
}

BEHAVIOR_CANDIDATES = [
    "session_duration",
    "input_timing_consistency",
    "keyboard_input_speed",
    "app_switching_frequency",
    "screen_active_time",
    "input_pause_patterns",
    "pin_entry_speed",
    "authentication_attempts",
    "handle_similarity_score",
]


def _drop_leaky_behavior_features(train_df: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, str]]:
    y = train_df[TARGET_COL].astype(int).to_numpy()
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for feature in features:
        lower_name = feature.strip().lower()
        if lower_name in LEAKAGE_FEATURE_ALIASES:
            dropped[feature] = "explicit_leakage_blocklist"
            continue

        vals = pd.to_numeric(train_df[feature], errors="coerce").fillna(0.0)
        if vals.nunique(dropna=True) <= 1:
            dropped[feature] = "constant_feature"
            continue

        normal_vals = vals[y == 0]
        fraud_vals = vals[y == 1]
        normal_std = float(normal_vals.std(ddof=0))
        mean_gap = float(abs(fraud_vals.mean() - normal_vals.mean()))
        if normal_std < 1e-6 and mean_gap > 0.05:
            dropped[feature] = "near_constant_normal_class"
            continue

        try:
            auc = float(roc_auc_score(y, vals.to_numpy(dtype=np.float64)))
            if auc >= 0.995 or auc <= 0.005:
                dropped[feature] = f"single_feature_auc={auc:.4f}"
                continue
        except Exception:
            pass

        kept.append(feature)
    return kept, dropped


def prepare_dataset2_splits_and_exports(dataset_path: Path, export_path: Path) -> dict[str, Any]:
    raw = pd.read_csv(dataset_path)
    graph_cols = ["user_id", "merchant_id", "amount", TARGET_COL]
    graph_df = raw[[c for c in graph_cols if c in raw.columns]].copy()

    cols = [c for c in KEEP_COLS if c in raw.columns]
    df = raw[cols].copy()
    df = df.drop_duplicates(subset=["transaction_id"])
    for c in ["pin_entry_method", "authorization_method", "transaction_type"]:
        if c in df.columns:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COL])
    numeric_cols = [c for c in df.columns if c not in ["transaction_id", "user_id", TARGET_COL]]
    # Ensure median imputation only sees numeric values; encode categorical leftovers.
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            train_codes, categories = pd.factorize(train_df[col], sort=True)
            mapping = {val: idx for idx, val in enumerate(categories)}
            train_df.loc[:, col] = pd.Series(train_codes, index=train_df.index, dtype="float64").replace(-1, np.nan)
            test_df.loc[:, col] = (
                test_df[col].map(mapping).astype("float64", copy=False).fillna(np.nan)
            )

    imputer = SimpleImputer(strategy="median")
    train_df.loc[:, numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df.loc[:, numeric_cols] = imputer.transform(test_df[numeric_cols])
    scaler = MinMaxScaler()
    train_df.loc[:, numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df.loc[:, numeric_cols] = scaler.transform(test_df[numeric_cols])

    export_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(export_path, index=False)
    stats = {
        "train": train_df[TARGET_COL].value_counts(normalize=True).to_dict(),
        "test": test_df[TARGET_COL].value_counts(normalize=True).to_dict(),
    }
    return {
        "train": train_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
        "imputer": imputer,
        "scaler": scaler,
        "stats": stats,
        "graph_df": graph_df,
    }


def train_and_export_l3(dataset_path: Path, models_dir: Path, export_path: Path) -> dict[str, float]:
    split = prepare_dataset2_splits_and_exports(dataset_path, export_path)
    train_df, test_df = split["train"], split["test"]
    graph_df = split["graph_df"]
    behavior_features = [c for c in BEHAVIOR_CANDIDATES if c in train_df.columns]
    behavior_features, dropped_features = _drop_leaky_behavior_features(train_df, behavior_features)
    if not behavior_features:
        raise ValueError("No behavioral features available for L3 autoencoder training.")

    normal_train = train_df[train_df[TARGET_COL].astype(int) == 0]
    if normal_train.empty:
        raise ValueError("No non-fraud rows available in training split for L3 autoencoder.")
    behavior_scaler = MinMaxScaler()
    X_train = behavior_scaler.fit_transform(normal_train[behavior_features].to_numpy(dtype=np.float32))
    X_test = behavior_scaler.transform(test_df[behavior_features].to_numpy(dtype=np.float32))
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    model = LSTMAutoencoder(n_features=X_train.shape[1], hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_loader = torch.utils.data.DataLoader(torch.tensor(X_train).unsqueeze(1), batch_size=64, shuffle=True)
    privacy_engine = None
    if PrivacyEngine is not None:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=0.5,
            target_delta=1e-5,
            epochs=5,
            max_grad_norm=1.0,
        )
    model.train()
    for _ in range(5):
        for batch in data_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        rec_train = model(torch.tensor(X_train).unsqueeze(1))
        err_train = torch.mean((torch.tensor(X_train).unsqueeze(1) - rec_train) ** 2, dim=(1, 2)).numpy()
        rec_test = model(torch.tensor(X_test).unsqueeze(1))
        err_test = torch.mean((torch.tensor(X_test).unsqueeze(1) - rec_test) ** 2, dim=(1, 2)).numpy()
    threshold = float(np.percentile(err_train, 99.0))
    pred = (err_test >= threshold).astype(int)
    normal_mask = y_test == 0
    fraud_mask = y_test == 1
    normal_err_mean = float(np.mean(err_test[normal_mask])) if np.any(normal_mask) else 0.0
    fraud_err_mean = float(np.mean(err_test[fraud_mask])) if np.any(fraud_mask) else 0.0
    reconstruction_gap_ratio = float(fraud_err_mean / max(normal_err_mean, 1e-6))
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f2 = fbeta_score(y_test, pred, beta=2, zero_division=0)
    auc_pr = average_precision_score(y_test, err_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)

    models_dir.mkdir(parents=True, exist_ok=True)
    graph_meta = {"method": "networkx_fallback", "pagerank_mean": 0.0}
    if HAS_PYG and "user_id" in graph_df.columns and "merchant_id" in graph_df.columns:
        graph_df = graph_df.dropna(subset=["user_id", "merchant_id"]).copy()
        graph_df["amount"] = pd.to_numeric(graph_df["amount"], errors="coerce").fillna(0.0)
        graph_df[TARGET_COL] = pd.to_numeric(graph_df[TARGET_COL], errors="coerce").fillna(0.0)

        users = graph_df["user_id"].astype(str).unique().tolist()
        merchants = graph_df["merchant_id"].astype(str).unique().tolist()
        nodes = users + merchants
        idx_map = {n: i for i, n in enumerate(nodes)}
        src = graph_df["user_id"].astype(str).map(idx_map).to_numpy(dtype=np.int64)
        dst = graph_df["merchant_id"].astype(str).map(idx_map).to_numpy(dtype=np.int64)
        edge_index = torch.tensor(
            np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
            dtype=torch.long,
        )

        x = np.zeros((len(nodes), 4), dtype=np.float32)
        y_node = np.zeros((len(nodes),), dtype=np.float32)

        user_agg = (
            graph_df.groupby(graph_df["user_id"].astype(str))
            .agg(txn_count=("amount", "count"), avg_amount=("amount", "mean"), fraud_rate=(TARGET_COL, "mean"))
            .fillna(0.0)
        )
        merchant_agg = (
            graph_df.groupby(graph_df["merchant_id"].astype(str))
            .agg(txn_count=("amount", "count"), avg_amount=("amount", "mean"), fraud_rate=(TARGET_COL, "mean"))
            .fillna(0.0)
        )

        max_user_txn = float(max(user_agg["txn_count"].max() if not user_agg.empty else 1.0, 1.0))
        max_merchant_txn = float(max(merchant_agg["txn_count"].max() if not merchant_agg.empty else 1.0, 1.0))
        max_amount = float(max(graph_df["amount"].max(), 1.0))

        for user in users:
            idx = idx_map[user]
            row = user_agg.loc[user] if user in user_agg.index else None
            x[idx, 0] = 1.0
            if row is not None:
                x[idx, 1] = float(row["txn_count"]) / max_user_txn
                x[idx, 2] = float(row["avg_amount"]) / max_amount
                x[idx, 3] = float(row["fraud_rate"])
                y_node[idx] = 1.0 if float(row["fraud_rate"]) >= 0.10 else 0.0

        for merchant in merchants:
            idx = idx_map[merchant]
            row = merchant_agg.loc[merchant] if merchant in merchant_agg.index else None
            x[idx, 0] = 0.0
            if row is not None:
                x[idx, 1] = float(row["txn_count"]) / max_merchant_txn
                x[idx, 2] = float(row["avg_amount"]) / max_amount
                x[idx, 3] = float(row["fraud_rate"])
                y_node[idx] = 1.0 if float(row["fraud_rate"]) >= 0.10 else 0.0

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y_node, dtype=torch.float32)
        gmodel = GraphSAGEModel(in_dim=4, hidden_dim=32, out_dim=16)
        optim_g = torch.optim.Adam(gmodel.parameters(), lr=1e-3)
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

        for _ in range(30):
            optim_g.zero_grad()
            emb = gmodel(data.x, data.edge_index)
            logits = gmodel.classify(emb)
            loss = F.binary_cross_entropy_with_logits(logits, data.y)
            loss.backward()
            optim_g.step()
        emb = gmodel(data.x, data.edge_index).detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1)
        norm_mean = float(np.mean(norms))
        norm_std = float(np.std(norms) + 1e-6)

        torch.save(gmodel.state_dict(), models_dir / "l3_graphsage.pth")
        np.savez(models_dir / "l3_graphsage_embeddings.npz", node_ids=np.asarray(nodes, dtype=str), embeddings=emb)
        graph_meta = {
            "method": "graphsage_pyg",
            "node_count": len(nodes),
            "embedding_dim": int(emb.shape[1]),
            "embedding_norm_mean": norm_mean,
            "embedding_norm_std": norm_std,
        }

    torch.save(model._module.state_dict() if hasattr(model, "_module") else model.state_dict(), models_dir / "l3_autoencoder.pth")
    with open(models_dir / "l3_scaler.pkl", "wb") as f:
        pickle.dump(behavior_scaler, f)

    meta = {
        "reconstruction_threshold": threshold,
        "behavior_features": behavior_features,
        "dropped_behavior_features": dropped_features,
        "normal_reconstruction_error_mean": normal_err_mean,
        "fraud_reconstruction_error_mean": fraud_err_mean,
        "reconstruction_gap_ratio": reconstruction_gap_ratio,
        "dp_target_epsilon": 0.5,
        "dp_actual_epsilon": float(privacy_engine.get_epsilon(1e-5)) if privacy_engine is not None else None,
        "graph_meta": graph_meta,
    }
    (models_dir / "l3_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    provenance_rows = [
        {
            "source_dataset": str(dataset_path),
            "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
            "label_origin": TARGET_COL,
            "synthetic_flag": 0,
            "synthetic_method": "",
            "model_version": settings.model_version,
        }
    ]
    pd.DataFrame(provenance_rows).to_csv(models_dir / "provenance_dataset2.csv", index=False)

    if mlflow is not None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("Kavach.ai-L3")
        with mlflow.start_run(run_name=f"{settings.model_version}-dataset2"):
            mlflow.log_params({"dp_target_epsilon": 0.5, "graph_method": graph_meta["method"]})
            mlflow.log_metrics(
                {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f2": float(f2),
                    "auc_pr": float(auc_pr),
                    "fpr": float(fpr),
                    "reconstruction_gap_ratio": reconstruction_gap_ratio,
                }
            )
            mlflow.log_artifact(str(export_path))
            mlflow.log_artifact(str(models_dir / "l3_metadata.json"))
            if (models_dir / "l3_graphsage_embeddings.npz").exists():
                mlflow.log_artifact(str(models_dir / "l3_graphsage_embeddings.npz"))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "auc_pr": float(auc_pr),
        "fpr": float(fpr),
        "reconstruction_gap_ratio": reconstruction_gap_ratio,
    }


def main() -> None:
    split = prepare_dataset2_splits_and_exports(settings.dataset2_path, settings.dataset2_export_path)
    print("[Dataset2] train distribution:", split["stats"]["train"])
    print("[Dataset2] test distribution:", split["stats"]["test"])
    metrics = train_and_export_l3(settings.dataset2_path, settings.models_dir, settings.dataset2_export_path)
    print("[Dataset2] metrics:", metrics)


if __name__ == "__main__":
    main()
