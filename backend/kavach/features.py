from __future__ import annotations

import asyncio
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from redis import Redis

from kavach.config import settings


def _to_unix(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


@dataclass
class FeatureComputer:
    redis: Redis
    user_amount_history: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=2000)))
    user_hour_history: dict[str, deque[int]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=2000)))
    device_seen: set[str] = field(default_factory=set)
    device_count: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    user_merchant_history: dict[str, deque[str]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=2000)))

    def _redis_key(self, user_id: str) -> str:
        return f"kavach:features:user:{user_id}"

    def _velocity_key(self, user_id: str, window: str) -> str:
        return f"kavach:velocity:{window}:{user_id}"

    def _device_last_key(self, device_type: str) -> str:
        return f"kavach:last_ts:device:{device_type}"

    def _update_velocity(self, user_id: str, amount: float, merchant: str, ts_unix: float) -> dict[str, float]:
        windows = {"1m": 60, "5m": 300, "1h": 3600}
        out: dict[str, float] = {}
        pipe = self.redis.pipeline()
        for label, sec in windows.items():
            key = self._velocity_key(user_id, label)
            pipe.zremrangebyscore(key, 0, ts_unix - sec)
            pipe.zadd(key, {f"{ts_unix}": ts_unix})
            pipe.zcard(key)
            pipe.expire(key, sec + 10)
        pipe.execute()

        for label in windows:
            key = self._velocity_key(user_id, label)
            out[f"txn_count_{label}"] = float(self.redis.zcard(key))

        amt_key = f"kavach:amount_sum_1h:{user_id}"
        self.redis.zremrangebyscore(amt_key, 0, ts_unix - 3600)
        self.redis.zadd(amt_key, {f"{ts_unix}:{amount:.2f}": ts_unix})
        self.redis.expire(amt_key, 3610)
        values = self.redis.zrange(amt_key, 0, -1)
        amount_sum_1h = 0.0
        for raw in values:
            try:
                amount_sum_1h += float(raw.decode("utf-8").split(":")[1])
            except Exception:
                continue
        out["amount_sum_1h"] = amount_sum_1h

        merchant_key = f"kavach:merchant_1h:{user_id}"
        self.redis.zremrangebyscore(merchant_key, 0, ts_unix - 3600)
        self.redis.zadd(merchant_key, {merchant: ts_unix})
        self.redis.expire(merchant_key, 3610)
        out["unique_merchant_1h"] = float(self.redis.zcard(merchant_key))
        return out

    def ingest_transaction(self, txn: dict[str, Any]) -> dict[str, float]:
        user_id = str(txn.get("user_id") or txn.get("transaction_id"))
        device_type = str(txn.get("device_type", "unknown"))
        merchant = str(txn.get("merchant_category", "unknown"))
        amount = float(txn.get("amount", 0.0))
        ts = txn.get("timestamp")
        if isinstance(ts, str):
            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, datetime):
            ts_dt = ts
        else:
            ts_dt = datetime.now(timezone.utc)
        ts_unix = _to_unix(ts_dt)

        velocity = self._update_velocity(user_id, amount, merchant, ts_unix)

        self.user_amount_history[user_id].append(amount)
        self.user_hour_history[user_id].append(int(ts_dt.hour))
        self.user_merchant_history[user_id].append(merchant)

        mean = sum(self.user_amount_history[user_id]) / max(len(self.user_amount_history[user_id]), 1)
        var = sum((x - mean) ** 2 for x in self.user_amount_history[user_id]) / max(len(self.user_amount_history[user_id]), 1)
        std = math.sqrt(var) if var > 1e-8 else 1.0
        amount_z = (amount - mean) / std

        hours = list(self.user_hour_history[user_id])
        hour_percentile = sum(1 for h in hours if h <= ts_dt.hour) / max(len(hours), 1)

        is_new_device = 0.0 if device_type in self.device_seen else 1.0
        self.device_seen.add(device_type)
        self.device_count[device_type] += 1

        mcc = float(txn.get("merchant_category_code") or 0.0)
        mcc_risk = 1.0 + (mcc % 5)
        cross_category_anomaly = 1.0 if len(set(self.user_merchant_history[user_id])) > 5 and merchant not in set(list(self.user_merchant_history[user_id])[:-1]) else 0.0

        feature_snapshot = {
            **velocity,
            "amount_zscore_30d": float(amount_z),
            "hour_percentile_user": float(hour_percentile),
            "is_new_device": float(is_new_device),
            "device_txn_count": float(self.device_count[device_type]),
            "geographic_disparity": float(txn.get("geographic_disparity") or 0.0),
            "unusual_location_flag": float(txn.get("unusual_location_flag") or 0.0),
            "geographic_location_vs_ip": float(txn.get("geographic_location_vs_ip") or 0.0),
            "merchant_category_risk_tier": float(mcc_risk),
            "cross_category_anomaly": float(cross_category_anomaly),
            "hour_sin": float(math.sin(2 * math.pi * ts_dt.hour / 24.0)),
            "hour_cos": float(math.cos(2 * math.pi * ts_dt.hour / 24.0)),
            "is_weekend": float(ts_dt.weekday() >= 5),
            "day_of_week": float(ts_dt.weekday()),
            "input_timing_consistency": float(txn.get("input_timing_consistency") or 0.0),
            "keyboard_input_speed": float(txn.get("keyboard_input_speed") or 0.0),
            "app_switching_frequency": float(txn.get("app_switching_frequency") or 0.0),
            "screen_active_time": float(txn.get("screen_active_time") or 0.0),
            "time_between_otp_generation_and_input": float(txn.get("time_between_otp_generation_and_input") or 0.0),
            "pin_entry_speed": float(txn.get("pin_entry_speed") or 0.0),
            "otp_request_frequency": float(txn.get("otp_request_frequency") or 0.0),
            "transaction_velocity": float(txn.get("transaction_velocity") or 0.0),
            "failed_transaction_count": float(txn.get("failed_transaction_count") or 0.0),
        }

        self.redis.hset(self._redis_key(user_id), mapping={k: str(v) for k, v in feature_snapshot.items()})
        self.redis.expire(self._redis_key(user_id), 31 * 24 * 3600)
        self.redis.set(self._device_last_key(device_type), ts_unix, ex=24 * 3600)
        return feature_snapshot

    def read_snapshot(self, txn: dict[str, Any]) -> dict[str, float]:
        user_id = str(txn.get("user_id") or txn.get("transaction_id"))
        raw = self.redis.hgetall(self._redis_key(user_id))
        if not raw:
            return {
                "txn_count_1m": 0.0,
                "txn_count_5m": 0.0,
                "txn_count_1h": 0.0,
                "amount_sum_1h": 0.0,
                "unique_merchant_1h": 0.0,
                "amount_zscore_30d": 0.0,
                "hour_percentile_user": 0.0,
                "is_new_device": 1.0,
                "device_txn_count": 0.0,
                "geographic_disparity": float(txn.get("geographic_disparity") or 0.0),
                "unusual_location_flag": float(txn.get("unusual_location_flag") or 0.0),
                "geographic_location_vs_ip": float(txn.get("geographic_location_vs_ip") or 0.0),
                "merchant_category_risk_tier": 3.0,
                "cross_category_anomaly": 0.0,
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "is_weekend": 0.0,
                "day_of_week": 0.0,
                "input_timing_consistency": float(txn.get("input_timing_consistency") or 0.0),
                "keyboard_input_speed": float(txn.get("keyboard_input_speed") or 0.0),
                "app_switching_frequency": float(txn.get("app_switching_frequency") or 0.0),
                "screen_active_time": float(txn.get("screen_active_time") or 0.0),
                "time_between_otp_generation_and_input": float(txn.get("time_between_otp_generation_and_input") or 0.0),
                "pin_entry_speed": float(txn.get("pin_entry_speed") or 0.0),
                "otp_request_frequency": float(txn.get("otp_request_frequency") or 0.0),
                "transaction_velocity": float(txn.get("transaction_velocity") or 0.0),
                "failed_transaction_count": float(txn.get("failed_transaction_count") or 0.0),
            }
        return {k.decode("utf-8"): float(v) for k, v in raw.items()}

    def get_last_device_ts(self, device_type: str) -> float | None:
        raw = self.redis.get(self._device_last_key(device_type))
        if raw is None:
            return None
        return float(raw)


async def simulated_flink_feature_job(queue: asyncio.Queue, feature_store: FeatureComputer, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            txn = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        try:
            feature_store.ingest_transaction(txn)
        finally:
            queue.task_done()


def seed_features_from_dataset(dataset_path: str, feature_store: FeatureComputer, limit: int = 10000) -> None:
    df = pd.read_csv(dataset_path).head(limit)
    colmap = {
        "transaction id": "transaction_id",
        "transaction type": "transaction_type",
        "amount (INR)": "amount",
    }
    df = df.rename(columns=colmap)
    for record in df.to_dict(orient="records"):
        ts = record.get("timestamp")
        if ts:
            try:
                record["timestamp"] = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except Exception:
                record["timestamp"] = datetime.now(timezone.utc)
        else:
            record["timestamp"] = datetime.now(timezone.utc)
        if "user_id" not in record:
            record["user_id"] = str(record.get("transaction_id"))
        feature_store.ingest_transaction(record)

