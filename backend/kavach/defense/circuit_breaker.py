from __future__ import annotations

from collections import deque
import statistics

from kavach.config import settings


class LatencyCircuitBreaker:
    def __init__(self) -> None:
        self.samples_ms: deque[float] = deque(maxlen=3000)
        self.fallback_mode: bool = False
        self.events: list[dict] = []

    def record(self, latency_ms: float) -> None:
        self.samples_ms.append(latency_ms)
        if len(self.samples_ms) < 30:
            return
        if len(self.samples_ms) < 100:
            p99 = max(self.samples_ms)
        else:
            p99 = statistics.quantiles(list(self.samples_ms), n=100)[98]
        if not self.fallback_mode and p99 > settings.circuit_breaker_trip_p99_ms:
            self.fallback_mode = True
            self.events.append({"event": "TRIP", "p99_ms": float(p99)})
        elif self.fallback_mode and p99 < settings.circuit_breaker_recover_p99_ms:
            self.fallback_mode = False
            self.events.append({"event": "RECOVER", "p99_ms": float(p99)})

