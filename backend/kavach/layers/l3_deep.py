from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        hidden_1 = max(8, n_features + 4)
        hidden_2 = max(4, n_features // 2 + 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class L3DeepDetector:
    models_dir: Path
    autoencoder: nn.Module | None = None
    scaler: MinMaxScaler | None = None
    threshold: float = 1.0
    behavior_features: list[str] = field(
        default_factory=lambda: [
            "session_duration",
            "input_timing_consistency",
            "keyboard_input_speed",
            "app_switching_frequency",
            "screen_active_time",
            "input_pause_patterns",
            "time_between_otp_generation_and_input",
            "pin_entry_speed",
        ]
    )
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    pagerank_baseline: dict[str, float] = field(default_factory=dict)
    graphsage_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    embedding_norm_mean: float = 0.0
    embedding_norm_std: float = 1.0
    autoencoder_type: str = "lstm"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def load(self) -> None:
        meta_path = self.models_dir / "l3_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.threshold = float(meta.get("reconstruction_threshold", self.threshold))
            self.behavior_features = meta.get("behavior_features", self.behavior_features)
            self.autoencoder_type = str(meta.get("autoencoder_type", self.autoencoder_type))
            graph_meta = meta.get("graph_meta", {})
            self.embedding_norm_mean = float(graph_meta.get("embedding_norm_mean", 0.0))
            self.embedding_norm_std = float(max(graph_meta.get("embedding_norm_std", 1.0), 1e-6))
        scaler_path = self.models_dir / "l3_scaler.pkl"
        if scaler_path.exists():
            import pickle

            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                try:
                    import joblib

                    self.scaler = joblib.load(scaler_path)
                except Exception:
                    self.scaler = None
        emb_path = self.models_dir / "l3_graphsage_embeddings.npz"
        if emb_path.exists():
            payload = np.load(emb_path)
            node_ids = payload["node_ids"]
            embs = payload["embeddings"]
            self.graphsage_embeddings = {str(node_ids[i]): embs[i] for i in range(len(node_ids))}
        model_path = self.models_dir / "l3_autoencoder.pth"
        input_dim = len(self.behavior_features)
        if self.autoencoder_type == "dense_v1":
            self.autoencoder = DenseAutoencoder(input_dim)
        else:
            self.autoencoder = LSTMAutoencoder(input_dim, hidden_dim=32)
        if model_path.exists():
            state = torch.load(model_path, map_location="cpu")
            self.autoencoder.load_state_dict(state)
        self.autoencoder.eval()

    def _graphsage_embedding_anomaly(self, user: str, merchant: str) -> float:
        if not self.graphsage_embeddings:
            return 0.0
        user_emb = self.graphsage_embeddings.get(user)
        merchant_emb = self.graphsage_embeddings.get(merchant)
        if user_emb is None and merchant_emb is None:
            return 0.5

        scores: list[float] = []
        for emb in [user_emb, merchant_emb]:
            if emb is None:
                continue
            norm = float(np.linalg.norm(emb))
            z = abs(norm - self.embedding_norm_mean) / max(self.embedding_norm_std, 1e-6)
            scores.append(float(min(1.0, z / 6.0)))

        if user_emb is not None and merchant_emb is not None:
            den = float(np.linalg.norm(user_emb) * np.linalg.norm(merchant_emb) + 1e-6)
            cos_sim = float(np.dot(user_emb, merchant_emb) / den)
            cos_anom = (1.0 - max(-1.0, min(1.0, cos_sim))) / 2.0
            scores.append(float(max(0.0, min(1.0, cos_anom))))

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _behavior_vector(self, txn: dict[str, Any]) -> np.ndarray:
        vec = np.asarray([[float(txn.get(f, 0.0) or 0.0) for f in self.behavior_features]], dtype=np.float32)
        if self.scaler is not None:
            vec = self.scaler.transform(vec)
        return vec

    def _reconstruction_error(self, txn: dict[str, Any]) -> float:
        if self.autoencoder is None:
            return 0.0
        vec = self._behavior_vector(txn)
        tensor = torch.tensor(vec, dtype=torch.float32)
        with torch.no_grad():
            if self.autoencoder_type == "dense_v1":
                reconstructed = self.autoencoder(tensor)
                error = torch.mean((tensor - reconstructed) ** 2).item()
            else:
                seq_tensor = tensor.unsqueeze(1)
                reconstructed = self.autoencoder(seq_tensor)
                error = torch.mean((seq_tensor - reconstructed) ** 2).item()
        return float(error)

    def _graph_anomaly(self, txn: dict[str, Any]) -> float:
        user = str(txn.get("user_id") or txn.get("transaction_id"))
        merchant = str(txn.get("merchant_id") or txn.get("merchant_category"))
        amount = float(txn.get("amount", 0.0))
        self.graph.add_node(user, kind="user")
        self.graph.add_node(merchant, kind="merchant")
        self.graph.add_edge(user, merchant, weight=amount)

        if self.graph.number_of_nodes() < 10:
            return 0.0
        pr = nx.pagerank(self.graph, alpha=0.85)
        if not self.pagerank_baseline:
            self.pagerank_baseline = pr
            return 0.0
        deltas = []
        for node, value in pr.items():
            base = self.pagerank_baseline.get(node, value)
            deltas.append(abs(value - base))
        score = float(np.percentile(np.asarray(deltas, dtype=np.float32), 95))
        try:
            cluster = nx.clustering(self.graph.to_undirected(), user)
        except Exception:
            cluster = 0.0
        try:
            bet = nx.betweenness_centrality(self.graph, k=min(20, self.graph.number_of_nodes()))
            bet_user = float(bet.get(user, 0.0))
        except Exception:
            bet_user = 0.0
        centrality_anomaly = float(min(1.0, score + abs(cluster - 0.05) + 2.0 * bet_user))
        embedding_anomaly = self._graphsage_embedding_anomaly(user, merchant)
        anomaly = 0.6 * centrality_anomaly + 0.4 * embedding_anomaly
        return float(min(1.0, anomaly))

    async def enrich(self, txn: dict[str, Any]) -> dict[str, float]:
        async with self.lock:
            rec_error = self._reconstruction_error(txn)
            graph_score = self._graph_anomaly(txn)
        return {
            "reconstruction_error": rec_error,
            "reconstruction_is_anomalous": 1.0 if rec_error >= self.threshold else 0.0,
            "graph_anomaly_score": graph_score,
        }
