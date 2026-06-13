import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET_COL = "is_fraud"
LEAKAGE_FEATURE_ALIASES = {"d1", "d_1", "feature_d1", "time_between_otp_generation_and_input"}
BASE_FEATURES = [
    "keyboard_input_speed",
    "input_timing_consistency",
    "input_pause_patterns",
    "app_switching_frequency",
    "pin_entry_speed",
    "session_duration",
    "authentication_attempts",
    "handle_similarity_score",
    "screen_active_time",
]


def _drop_leaky_features(train_df: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, str]]:
    y = train_df[TARGET_COL].astype(int).to_numpy()
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for feature in features:
        feature_l = feature.strip().lower()
        if feature_l in LEAKAGE_FEATURE_ALIASES:
            dropped[feature] = "explicit_leakage_blocklist"
            continue
        vals = pd.to_numeric(train_df[feature], errors="coerce").fillna(0.0)
        if vals.nunique(dropna=True) <= 1:
            dropped[feature] = "constant_feature"
            continue
        normal_vals = vals[y == 0]
        fraud_vals = vals[y == 1]
        if float(normal_vals.std(ddof=0)) < 1e-6 and float(abs(fraud_vals.mean() - normal_vals.mean())) > 0.05:
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


class KavachAutoencoder(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, max(8, n_features + 4)),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(max(8, n_features + 4), max(4, n_features // 2 + 2)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(4, n_features // 2 + 2), max(8, n_features + 4)),
            nn.ReLU(),
            nn.Linear(max(8, n_features + 4), n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv("data/dataset_3.csv")
    features = [c for c in BASE_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No L3 features found in dataset_3.csv")

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL].astype(int),
    )

    features, dropped = _drop_leaky_features(train_df, features)
    if not features:
        raise ValueError("All candidate L3 features were dropped by leakage checks.")

    for col in features:
        med = float(train_df[col].median()) if np.isfinite(train_df[col].median()) else 0.0
        train_df.loc[:, col] = train_df[col].fillna(med)
        test_df.loc[:, col] = test_df[col].fillna(med)

    scaler = StandardScaler()
    train_normal = train_df[train_df[TARGET_COL].astype(int) == 0][features]
    if train_normal.empty:
        raise ValueError("No non-fraud rows in training split.")
    X_train = scaler.fit_transform(train_normal.to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[features].to_numpy(dtype=np.float32))
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, "models/l3_scaler.pkl")

    model = KavachAutoencoder(X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=128, shuffle=True)

    print("Training NPCI Behavioral Model...")
    for epoch in range(35):
        model.train()
        running = 0.0
        for (inputs,) in train_loader:
            noisy_inputs = inputs + 0.03 * torch.randn_like(inputs)
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Avg Loss: {running / max(len(train_loader), 1):.6f}")

    model.eval()
    with torch.no_grad():
        train_tensor = torch.FloatTensor(X_train)
        train_recon = model(train_tensor)
        train_err = criterion(train_recon, train_tensor).mean(dim=1).numpy()

        test_tensor = torch.FloatTensor(X_test)
        test_recon = model(test_tensor)
        test_err = criterion(test_recon, test_tensor).mean(dim=1).numpy()

    threshold = float(np.percentile(train_err, 99))
    normal_err = float(np.mean(test_err[y_test == 0])) if np.any(y_test == 0) else 0.0
    fraud_err = float(np.mean(test_err[y_test == 1])) if np.any(y_test == 1) else 0.0
    gap_ratio = float(fraud_err / max(normal_err, 1e-6))
    flagged_rate = float(np.mean(test_err[y_test == 0] >= threshold)) if np.any(y_test == 0) else 0.0

    print("\n--- LEAKAGE-AUDITED VALIDATION ---")
    print(f"Features used ({len(features)}): {features}")
    print(f"Dropped features: {dropped if dropped else 'None'}")
    print(f"Threshold (P99 train-normal): {threshold:.6f}")
    print(f"Normal Error (MSE): {normal_err:.6f}")
    print(f"Fraud Error (MSE): {fraud_err:.6f}")
    print(f"Gap Ratio: {gap_ratio:.2f}x")
    print(f"Normal false positive rate vs threshold: {flagged_rate * 100:.2f}%")

    torch.save(model.state_dict(), "models/l3_autoencoder.pth")
    metadata = {
        "autoencoder_type": "dense_v1",
        "behavior_features": features,
        "dropped_behavior_features": dropped,
        "reconstruction_threshold": threshold,
        "normal_reconstruction_error_mean": normal_err,
        "fraud_reconstruction_error_mean": fraud_err,
        "reconstruction_gap_ratio": gap_ratio,
    }
    Path("models/l3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
