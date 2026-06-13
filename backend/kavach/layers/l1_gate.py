from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from redis import Redis

from kavach.config import settings


@dataclass
class L1GateResult:
    decision: str
    reason: str
    model_score: float


@dataclass
class L1Gate:
    redis: Redis
    model_path: Path
    session: ort.InferenceSession | None = None
    input_name: str | None = None
    output_name: str | None = None
    blocklist_bank_pairs: set[str] = field(default_factory=set)

    def load(self) -> None:
        self.blocklist_bank_pairs = {"BADBANK_A|MULEBANK_X", "RISKYBANK_77|RISKYBANK_88"}
        if self.model_path.exists():
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def add_block_rule(self, sender_bank: str, receiver_bank: str) -> None:
        key = f"{sender_bank}|{receiver_bank}"
        self.blocklist_bank_pairs.add(key)
        self.redis.sadd("kavach:blocklist:banks", key)

    def _last_device_key(self, device_fingerprint: str) -> str:
        return f"kavach:l1:last_device_ts:{device_fingerprint}"

    def _velocity_minute_key(self, user_id: str, minute_bucket: int) -> str:
        return f"kavach:l1:velocity:{user_id}:{minute_bucket}"

    def _rolling_velocity_1m(self, user_id: str, now_dt: datetime) -> int:
        now_unix = int(now_dt.timestamp())
        curr_minute = now_unix // 60
        curr_second = now_unix % 60
        curr_key = self._velocity_minute_key(user_id, curr_minute)
        prev_key = self._velocity_minute_key(user_id, curr_minute - 1)

        # Redis BITFIELD stores per-second counters (u8) for a rolling 60-second window.
        bump = self.redis.pipeline()
        bump.execute_command("BITFIELD", curr_key, "OVERFLOW", "SAT", "INCRBY", "u8", curr_second * 8, 1)
        bump.expire(curr_key, 180)
        bump.execute()

        read = self.redis.pipeline()
        for sec in range(curr_second + 1):
            read.execute_command("BITFIELD", curr_key, "GET", "u8", sec * 8)
        for sec in range(curr_second + 1, 60):
            read.execute_command("BITFIELD", prev_key, "GET", "u8", sec * 8)
        values = read.execute()

        total = 0
        for item in values:
            if not item:
                continue
            val = item[0]
            if val is not None:
                total += int(val)
        return total

    def _subsecond_bot(self, txn: dict[str, Any], user_id: str) -> bool:
        device_fingerprint = str(txn.get("device_id") or f"{user_id}:{txn.get('device_type', 'unknown')}")
        now_ts = txn.get("timestamp")
        if isinstance(now_ts, str):
            now_dt = datetime.fromisoformat(now_ts.replace("Z", "+00:00"))
        elif isinstance(now_ts, datetime):
            now_dt = now_ts
        else:
            now_dt = datetime.now(timezone.utc)
        now_unix = now_dt.timestamp()
        key = self._last_device_key(device_fingerprint)
        prev = self.redis.get(key)
        self.redis.set(key, now_unix, ex=300)
        if prev is None:
            return False
        return (now_unix - float(prev)) < float(settings.l1_subsecond_min_interval_sec)

    def _model_score(self, feature_vec: list[float]) -> float:
        if self.session and self.input_name and self.output_name:
            arr = np.asarray([feature_vec], dtype=np.float32)
            pred = self.session.run([self.output_name], {self.input_name: arr})[0]
            if pred.ndim == 2 and pred.shape[1] > 1:
                return float(pred[0, 1])
            return float(pred.ravel()[0])
        amount = feature_vec[0]
        velocity = feature_vec[1]
        anomaly = feature_vec[2]
        amount_z = feature_vec[3] if len(feature_vec) > 3 else 0.0
        score = min(
            1.0,
            0.03
            + 0.000002 * amount
            + 0.05 * velocity
            + 0.22 * anomaly
            + 0.12 * max(0.0, amount_z - 1.5),
        )
        return float(score)

    def evaluate(self, txn: dict[str, Any], features: dict[str, float]) -> L1GateResult:
        ts = txn.get("timestamp")
        if isinstance(ts, str):
            now_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, datetime):
            now_dt = ts
        else:
            now_dt = datetime.now(timezone.utc)
        user_id = str(txn.get("user_id") or txn.get("transaction_id"))

        pair_key = f"{txn.get('sender_bank')}|{txn.get('receiver_bank')}"
        if pair_key in self.blocklist_bank_pairs or self.redis.sismember("kavach:blocklist:banks", pair_key):
            return L1GateResult("BLOCK", "historical_blocklist", 1.0)

        amount = float(txn.get("amount", 0.0))
        if amount > settings.l1_amount_ceiling_inr:
            return L1GateResult("ESCALATE", "amount_ceiling", 0.95)

        txn_count_1m = self._rolling_velocity_1m(user_id, now_dt)
        if txn_count_1m > settings.l1_velocity_escalate_threshold:
            return L1GateResult("ESCALATE", "velocity_1m", 0.8)

        is_weekend = bool(float(features.get("is_weekend", 0.0)) >= 0.5)
        hour = int(getattr(txn.get("timestamp"), "hour", 0) if not isinstance(txn.get("timestamp"), str) else datetime.fromisoformat(txn.get("timestamp").replace("Z", "+00:00")).hour)
        if is_weekend and 1 <= hour <= 4 and amount > settings.weekend_night_amount_inr:
            return L1GateResult("ESCALATE", "weekend_late_night", 0.77)

        if self._subsecond_bot(txn, user_id):
            return L1GateResult("BLOCK", "subsecond_device_cluster", 0.99)

        feature_vec = [
            amount,
            float(features.get("txn_count_1m", 0.0)),
            float(features.get("is_new_device", 0.0)),
            float(features.get("amount_zscore_30d", 0.0)),
        ]
        model_score = self._model_score(feature_vec)
        if model_score >= settings.l1_xgb_threshold:
            return L1GateResult("ESCALATE", "l1_xgb_recall_gate", model_score)
        return L1GateResult("PASS", "clear_by_l1", model_score)
