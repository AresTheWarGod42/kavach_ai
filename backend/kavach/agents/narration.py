from __future__ import annotations

from typing import Any

import httpx

from kavach.config import settings


class NarrationAgent:
    def __init__(self) -> None:
        self.base_url = settings.flowise_url.rstrip("/")
        self.path = settings.flowise_prediction_path

    async def create_case_brief(self, payload: dict[str, Any]) -> str:
        data = {
            "question": (
                "Generate a concise analyst brief grounded only in provided structured data. "
                "Temperature 0. No speculation."
            ),
            "overrideConfig": {"temperature": 0},
            "inputs": payload,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(f"{self.base_url}{self.path}", json=data)
                if res.status_code < 300:
                    body = res.json()
                    if isinstance(body, dict):
                        for key in ["text", "answer", "output", "result"]:
                            if key in body and isinstance(body[key], str):
                                return body[key]
        except Exception:
            pass

        shap = payload.get("shap_top5", [])
        tier = payload.get("tier", "High")
        amount = payload.get("amount")
        sender_state = payload.get("sender_state")
        reasons = []
        for idx, s in enumerate(shap[:5], start=1):
            reasons.append(
                f"({idx}) {s.get('feature')}={s.get('feature_value')} [SHAP {s.get('shap_value', 0):+.2f}]"
            )
        reason_text = ", ".join(reasons) if reasons else "score crossed high-risk boundary."
        return (
            f"This transaction of INR {amount} from {sender_state} is flagged {tier} because {reason_text}. "
            f"L2 score: {payload.get('score'):.3f}. L3 reconstruction error: {payload.get('l3_reconstruction_error', 0):.4f}."
        )

