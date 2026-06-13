from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd

from kavach.config import settings


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "transaction id": "transaction_id",
        "transaction type": "transaction_type",
        "amount (INR)": "amount",
    }
    renamed = df.rename(columns=colmap)
    renamed.columns = [c.strip().replace(" ", "_") for c in renamed.columns]
    return renamed


@dataclass
class TransactionSimulator:
    dataset_path: str
    token: str
    tps: int = settings.simulator_tps
    fraud_injection_rate: float = settings.simulator_fraud_injection_rate
    base_url: str = settings.simulator_base_url
    running: bool = True

    def __post_init__(self) -> None:
        raw = pd.read_csv(self.dataset_path)
        self.df = _normalize_cols(raw)
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())
        self.seed_rows = self.df.to_dict(orient="records")
        self.user_ids = [str(r.get("transaction_id")) for r in self.seed_rows[:5000]]

    def _base_sample(self) -> dict[str, Any]:
        row = random.choice(self.seed_rows).copy()
        row["transaction_id"] = f"SIM-{uuid.uuid4().hex[:16]}"
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        row["user_id"] = random.choice(self.user_ids) if self.user_ids else row["transaction_id"]
        row["merchant_id"] = f"MER-{random.randint(1000, 9999)}"
        return row

    def _inject_fraud(self, txn: dict[str, Any]) -> dict[str, Any]:
        scenario = random.choice(["sim_swap", "velocity_attack", "mule_chain", "overnight_transfer"])
        if scenario == "sim_swap":
            txn["device_type"] = random.choice(["FeaturePhone", "UnknownEmulator", "NewAndroid"])
            txn["amount"] = max(float(txn.get("amount", 0.0)) * 4.5, 65000.0)
            txn["unusual_device_flag"] = 1
            txn["transaction_velocity"] = 6
        elif scenario == "velocity_attack":
            txn["amount"] = random.uniform(900, 1800)
            txn["transaction_velocity"] = 20
            txn["otp_request_frequency"] = random.uniform(7, 12)
            txn["failed_transaction_count"] = random.uniform(3, 7)
        elif scenario == "mule_chain":
            txn["sender_state"] = random.choice(["Rajasthan", "Delhi", "West Bengal", "Bihar", "Maharashtra"])
            txn["receiver_bank"] = random.choice(["MuleBankA", "MuleBankB", "MuleBankC"])
            txn["handle_similarity_score"] = random.uniform(0.8, 0.99)
            txn["geographic_disparity"] = random.uniform(0.85, 0.99)
        else:
            txn["amount"] = max(float(txn.get("amount", 0.0)), settings.overnight_large_transfer_inr)
            now = datetime.now(timezone.utc)
            forced = now.replace(hour=random.choice([1, 2, 3, 4]), minute=random.randint(0, 59), second=random.randint(0, 59))
            txn["timestamp"] = forced.isoformat()
            txn["is_weekend"] = 1
        return txn

    async def run(self) -> None:
        if self.tps <= 0:
            return
        interval = 1.0 / float(self.tps)
        headers = {"Authorization": f"Bearer {self.token}"}
        async with httpx.AsyncClient(timeout=3.0) as client:
            while self.running:
                txn = self._base_sample()
                if random.random() <= self.fraud_injection_rate:
                    txn = self._inject_fraud(txn)
                try:
                    await client.post(f"{self.base_url}/score", json=txn, headers=headers)
                except Exception:
                    pass
                await asyncio.sleep(interval)
