from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from fastapi import WebSocket


class WebSocketHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._stats = defaultdict(int)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
            self._stats["connected"] = len(self._clients)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
            self._stats["connected"] = len(self._clients)

    async def broadcast(self, payload: dict) -> None:
        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []
        async with self._lock:
            for ws in self._clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)
            self._stats["connected"] = len(self._clients)

    @property
    def connected_count(self) -> int:
        return self._stats["connected"]

