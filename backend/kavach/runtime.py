from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from kavach.config import settings
from kavach.ws import WebSocketHub


REQUEST_COUNT = Counter("kavach_request_total", "Total API requests", ["route", "method"])
REQUEST_LATENCY = Histogram("kavach_request_latency_seconds", "Request latency", ["route"])
FRAUD_COUNTER = Counter("kavach_fraud_flag_total", "Fraud transaction count", ["tier"])
CIRCUIT_BREAKER_GAUGE = Gauge("kavach_circuit_breaker_state", "Circuit breaker state")
LAYER_INVOCATIONS = Counter("kavach_layer_invocations_total", "Layer invocation count", ["layer"])


@dataclass
class RuntimeMetrics:
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    tx_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=30000))
    scored_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    score_window: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=50000))
    state_window: deque[tuple[float, str, bool]] = field(default_factory=lambda: deque(maxlen=50000))

    rolling_precision: float = 0.91
    rolling_recall: float = 0.86
    rolling_f2: float = 0.88
    rolling_auc_pr: float = 0.93
    rolling_fpr: float = 0.004
    drift_status: str = "stable"
    challenger_version: str | None = None
    shadow_mode: bool = False
    shadow_samples: int = 0
    shadow_disagreements: int = 0
    shadow_abs_deltas: deque[float] = field(default_factory=lambda: deque(maxlen=50000))
    shadow_champion_scores: deque[float] = field(default_factory=lambda: deque(maxlen=50000))
    shadow_challenger_scores: deque[float] = field(default_factory=lambda: deque(maxlen=50000))

    def track(self, latency_ms: float, score: float, state: str, is_fraud: bool) -> None:
        now = time.time()
        self.latencies_ms.append(latency_ms)
        self.tx_timestamps.append(now)
        self.score_window.append((now, score))
        self.state_window.append((now, state, is_fraud))

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        if len(self.latencies_ms) < 100:
            return float(max(self.latencies_ms))
        return float(statistics.quantiles(list(self.latencies_ms), n=100)[98])

    def tps(self, lookback_sec: int = 1) -> float:
        if not self.tx_timestamps:
            return 0.0
        cutoff = time.time() - lookback_sec
        count = sum(1 for t in self.tx_timestamps if t >= cutoff)
        return float(count / max(lookback_sec, 1))

    def fraud_rate(self, window_sec: int) -> float:
        if not self.state_window:
            return 0.0
        cutoff = time.time() - window_sec
        points = [p for p in self.state_window if p[0] >= cutoff]
        if not points:
            return 0.0
        fraud = sum(1 for _, _, is_fraud in points if is_fraud)
        return float(fraud / len(points))

    def score_histogram_bins(self) -> list[dict[str, float]]:
        cutoff = time.time() - 300
        scores = [s for t, s in self.score_window if t >= cutoff]
        bins: list[dict[str, float]] = []
        for idx in range(10):
            low = idx * 0.1
            high = low + 0.1
            count = sum(1 for s in scores if low <= s < high or (idx == 9 and s == 1.0))
            bins.append({"bucket": round(low, 1), "count": count})
        return bins

    def track_shadow(self, champion_score: float, challenger_score: float, champion_tier: str, challenger_tier: str) -> None:
        self.shadow_samples += 1
        self.shadow_champion_scores.append(float(champion_score))
        self.shadow_challenger_scores.append(float(challenger_score))
        self.shadow_abs_deltas.append(abs(float(champion_score) - float(challenger_score)))
        if champion_tier != challenger_tier:
            self.shadow_disagreements += 1

    def reset_shadow_tracking(self) -> None:
        self.shadow_samples = 0
        self.shadow_disagreements = 0
        self.shadow_abs_deltas.clear()
        self.shadow_champion_scores.clear()
        self.shadow_challenger_scores.clear()

    @property
    def shadow_disagreement_rate(self) -> float:
        if self.shadow_samples <= 0:
            return 0.0
        return float(self.shadow_disagreements / self.shadow_samples)

    @property
    def shadow_mean_abs_delta(self) -> float:
        if not self.shadow_abs_deltas:
            return 0.0
        return float(sum(self.shadow_abs_deltas) / len(self.shadow_abs_deltas))

    @property
    def shadow_champion_mean_score(self) -> float:
        if not self.shadow_champion_scores:
            return 0.0
        return float(sum(self.shadow_champion_scores) / len(self.shadow_champion_scores))

    @property
    def shadow_challenger_mean_score(self) -> float:
        if not self.shadow_challenger_scores:
            return 0.0
        return float(sum(self.shadow_challenger_scores) / len(self.shadow_challenger_scores))

    def state_hotspots(self, window_sec: int = 1800) -> dict[str, dict[str, float]]:
        cutoff = time.time() - window_sec
        out: dict[str, dict[str, float]] = {}
        for event in list(self.scored_events):
            ts = event.get("_unix_ts", 0.0)
            if ts < cutoff:
                continue
            state = str(event.get("sender_state", "Unknown"))
            slot = out.setdefault(state, {"fraud_count": 0.0, "total": 0.0, "fraud_rate": 0.0})
            slot["total"] += 1
            if event.get("tier") in ["High", "Critical"]:
                slot["fraud_count"] += 1
        for v in out.values():
            v["fraud_rate"] = v["fraud_count"] / max(v["total"], 1.0)
        return out


@dataclass
class KavachRuntime:
    redis: Any
    l1: Any
    l2: Any
    l3: Any
    feature_store: Any
    rate_limiter: Any
    audit_logger: Any
    circuit_breaker: Any
    drift_manager: Any
    narration_agent: Any
    rule_agent: Any
    threat_agent: Any
    ws_hub: WebSocketHub = field(default_factory=WebSocketHub)
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    alert_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    retrain_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    feature_stream_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    generated_token: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = settings.model_version
    shadow_l2: Any | None = None
    shadow_candidate_dir: str | None = None
    shadow_candidate_metrics: dict[str, float] = field(default_factory=dict)
    shadow_shap_ok: bool = True
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self.metrics.scored_events)

    def push_event(self, event: dict[str, Any]) -> None:
        event["_unix_ts"] = time.time()
        self.metrics.scored_events.appendleft(event)

    def model_metrics_payload(self) -> dict[str, Any]:
        return {
            "precision": self.metrics.rolling_precision,
            "recall": self.metrics.rolling_recall,
            "f2": self.metrics.rolling_f2,
            "auc_pr": self.metrics.rolling_auc_pr,
            "fpr": self.metrics.rolling_fpr,
            "drift_status": self.metrics.drift_status,
            "champion_version": self.model_version,
            "challenger_version": self.metrics.challenger_version,
            "shadow_mode": self.metrics.shadow_mode,
            "shadow_samples": self.metrics.shadow_samples,
            "shadow_disagreement_rate": self.metrics.shadow_disagreement_rate,
            "shadow_mean_abs_delta": self.metrics.shadow_mean_abs_delta,
            "shadow_champion_mean_score": self.metrics.shadow_champion_mean_score,
            "shadow_challenger_mean_score": self.metrics.shadow_challenger_mean_score,
        }

    def live_metrics_payload(self, active_alerts: int) -> dict[str, Any]:
        return {
            "tps": round(self.metrics.tps(1), 3),
            "p99_latency_ms": round(self.metrics.p99_latency_ms, 3),
            "fraud_rate_1m": round(self.metrics.fraud_rate(60), 4),
            "fraud_rate_5m": round(self.metrics.fraud_rate(300), 4),
            "fraud_rate_1h": round(self.metrics.fraud_rate(3600), 4),
            "active_alerts": active_alerts,
            "model_version": self.model_version,
            "circuit_breaker_status": "L1_ONLY" if self.circuit_breaker.fallback_mode else "L1_L2_ACTIVE",
        }


def tier_from_score(score: float) -> tuple[str, str]:
    if score < settings.score_threshold_medium:
        return "Low", "Allow - logged to baseline"
    if score < settings.score_threshold_high:
        return "Medium", "Allow + silent flag + enhanced logging"
    if score < settings.score_threshold_critical:
        return "High", "Step-up auth trigger + analyst alert queue"
    return "Critical", "Soft-block + user notification + mandatory analyst review"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
