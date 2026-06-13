from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from kavach.config import settings


class ThreatIntelAgent:
    def __init__(self, sources: list[str] | None = None) -> None:
        self.n8n_url = settings.n8n_url.rstrip("/")
        self.webhook_path = settings.threat_intel_webhook_path
        if sources is not None:
            self.sources = sources
        else:
            raw = settings.threat_intel_sources_csv.strip()
            self.sources = [s.strip() for s in raw.split(",") if s.strip()]

    def _normalize_updates(self, body: Any, source: str) -> list[dict[str, Any]]:
        rows = body if isinstance(body, list) else body.get("updates", []) if isinstance(body, dict) else []
        out: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            sender = item.get("sender_bank")
            receiver = item.get("receiver_bank")
            if not sender or not receiver:
                continue
            out.append(
                {
                    "sender_bank": str(sender),
                    "receiver_bank": str(receiver),
                    "source": source,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        return out

    async def poll(self) -> list[dict[str, Any]]:
        # Primary path: n8n orchestrates all threat-intel connectors.
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                res = await client.get(f"{self.n8n_url}{self.webhook_path}")
                if res.status_code < 300:
                    updates = self._normalize_updates(res.json(), "n8n")
                    if updates:
                        return updates
        except Exception:
            pass

        additions: list[dict[str, Any]] = []
        for source in self.sources:
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    res = await client.get(source)
                    if res.status_code < 300:
                        additions.extend(self._normalize_updates(res.json(), source))
            except Exception:
                continue

        if additions:
            return additions

        # Safe offline fallback for demo resilience when no feed is reachable.
        now = datetime.now(timezone.utc).isoformat()
        return [
            {"sender_bank": "RISKYBANK_77", "receiver_bank": "RISKYBANK_88", "source": "fallback-seed", "timestamp": now}
        ]
