from __future__ import annotations

from fastapi import Request

from kavach.runtime import KavachRuntime


def get_runtime(request: Request) -> KavachRuntime:
    return request.app.state.runtime

