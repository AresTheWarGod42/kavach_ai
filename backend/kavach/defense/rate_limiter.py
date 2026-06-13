from __future__ import annotations

import time
from dataclasses import dataclass

from redis import Redis

from kavach.config import settings


@dataclass
class RateLimitResult:
    allowed: bool
    count_last_minute: int
    force_high_tier: bool


class SlidingWindowRateLimiter:
    def __init__(self, redis_client: Redis) -> None:
        self.redis = redis_client

    def _key(self, user_id: str) -> str:
        return f"kavach:ratelimit:{user_id}"

    def check(self, user_id: str) -> RateLimitResult:
        now = time.time()
        key = self._key(user_id)
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - 60)
        pipe.zadd(key, {f"{now}": now})
        pipe.zcard(key)
        pipe.expire(key, 70)
        _, _, count, _ = pipe.execute()
        count = int(count)
        force_high = count > settings.throttle_limit_per_minute
        return RateLimitResult(allowed=True, count_last_minute=count, force_high_tier=force_high)

