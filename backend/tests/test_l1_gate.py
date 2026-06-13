from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kavach.layers.l1_gate import L1Gate


class FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def set(self, key: str, value: float, ex: int | None = None) -> None:
        del ex
        self._values[key] = str(value)

    def sadd(self, key: str, value: str) -> None:
        self._sets.setdefault(key, set()).add(value)

    def sismember(self, key: str, value: str) -> bool:
        return value in self._sets.get(key, set())


class L1GateSubsecondBotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis = FakeRedis()
        self.gate = L1Gate(redis=self.redis, model_path=Path("does-not-exist.onnx"))
        self.gate.load()
        self.base_time = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)
        self.base_features = {"txn_count_1m": 0.0, "is_weekend": 0.0, "is_new_device": 0.0}

    def _txn(self, user_id: str, timestamp: datetime) -> dict[str, object]:
        return {
            "transaction_id": f"TXN-{user_id}",
            "user_id": user_id,
            "timestamp": timestamp,
            "amount": 100.0,
            "sender_bank": "SAFEBANK_A",
            "receiver_bank": "SAFEBANK_B",
            "device_type": "Android",
        }

    def test_same_user_subsecond_is_blocked(self) -> None:
        first = self.gate.evaluate(self._txn("user-1", self.base_time), self.base_features)
        second = self.gate.evaluate(self._txn("user-1", self.base_time + timedelta(milliseconds=200)), self.base_features)
        self.assertEqual(first.decision, "PASS")
        self.assertEqual(second.decision, "BLOCK")

    def test_different_users_same_device_not_blocked(self) -> None:
        first = self.gate.evaluate(self._txn("user-a", self.base_time), self.base_features)
        second = self.gate.evaluate(self._txn("user-b", self.base_time + timedelta(milliseconds=200)), self.base_features)
        self.assertEqual(first.decision, "PASS")
        self.assertEqual(second.decision, "PASS")


if __name__ == "__main__":
    unittest.main()
