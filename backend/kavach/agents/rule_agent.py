from __future__ import annotations

from typing import Any

import httpx

from kavach.config import settings


class AdaptiveRuleAgent:
    def __init__(self) -> None:
        self.n8n_url = settings.n8n_url.rstrip("/")

    async def propose_rules(self, trigger_reason: str, cluster_summary: dict[str, Any]) -> list[dict[str, Any]]:
        payload = {"reason": trigger_reason, "false_negative_cluster": cluster_summary}
        try:
            async with httpx.AsyncClient(timeout=7.0) as client:
                res = await client.post(f"{self.n8n_url}/webhook/kavach-adaptive-rule", json=payload)
                if res.status_code < 300:
                    body = res.json()
                    if isinstance(body, list):
                        return body[:3]
                    if isinstance(body, dict) and isinstance(body.get("rules"), list):
                        return body["rules"][:3]
        except Exception:
            pass

        return [
            {
                "rule_name": "weekend_new_device_high_amount",
                "condition": "is_weekend==1 AND is_new_device==1 AND amount>25000",
                "expected_recall_gain": 0.03,
                "risk_of_fp": 0.01,
            },
            {
                "rule_name": "burst_multi_state_velocity",
                "condition": "txn_count_5m>8 AND cross_category_anomaly==1",
                "expected_recall_gain": 0.02,
                "risk_of_fp": 0.015,
            },
            {
                "rule_name": "otp_speed_anomaly",
                "condition": "pin_entry_speed<0.5 AND time_between_otp_generation_and_input<1.0",
                "expected_recall_gain": 0.015,
                "risk_of_fp": 0.009,
            },
        ]

