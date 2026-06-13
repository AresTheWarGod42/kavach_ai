from __future__ import annotations

from fastapi import APIRouter, Depends, Query, WebSocket
from fastapi.responses import JSONResponse

from kavach.dependencies import get_runtime
from kavach.runtime import KavachRuntime, REQUEST_COUNT
from kavach.schemas import AuditLogResponse, AuditRecordOut, LiveMetrics, ModelMetrics
from kavach.security import require_jwt, require_ws_jwt
from kavach.storage import AuditLogRecord, count_open_alerts, session_scope


router = APIRouter(tags=["metrics"])


@router.get("/metrics/live", response_model=LiveMetrics, dependencies=[Depends(require_jwt)])
def metrics_live(runtime: KavachRuntime = Depends(get_runtime)) -> LiveMetrics:
    REQUEST_COUNT.labels(route="/metrics/live", method="GET").inc()
    payload = runtime.live_metrics_payload(active_alerts=count_open_alerts())
    return LiveMetrics(**payload)


@router.get("/metrics/model", response_model=ModelMetrics, dependencies=[Depends(require_jwt)])
def metrics_model(runtime: KavachRuntime = Depends(get_runtime)) -> ModelMetrics:
    REQUEST_COUNT.labels(route="/metrics/model", method="GET").inc()
    payload = runtime.model_metrics_payload()
    return ModelMetrics(**payload)


@router.get("/metrics/distribution", dependencies=[Depends(require_jwt)])
def score_distribution(runtime: KavachRuntime = Depends(get_runtime)) -> JSONResponse:
    REQUEST_COUNT.labels(route="/metrics/distribution", method="GET").inc()
    return JSONResponse({"histogram": runtime.metrics.score_histogram_bins()})


@router.get("/metrics/hotspots", dependencies=[Depends(require_jwt)])
def fraud_hotspots(runtime: KavachRuntime = Depends(get_runtime)) -> JSONResponse:
    REQUEST_COUNT.labels(route="/metrics/hotspots", method="GET").inc()
    return JSONResponse({"states": runtime.metrics.state_hotspots()})


@router.get("/audit-log", response_model=AuditLogResponse, dependencies=[Depends(require_jwt)])
def get_audit_log(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    runtime: KavachRuntime = Depends(get_runtime),
) -> AuditLogResponse:
    REQUEST_COUNT.labels(route="/audit-log", method="GET").inc()
    chain_ok, _ = runtime.audit_logger.verify_chain()
    with session_scope() as session:
        total = session.query(AuditLogRecord).count()
        rows = (
            session.query(AuditLogRecord)
            .order_by(AuditLogRecord.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
    return AuditLogResponse(
        total=total,
        chain_valid=chain_ok,
        records=[
            AuditRecordOut(
                id=r.id,
                transaction_id=r.transaction_id,
                timestamp=r.timestamp,
                l1_decision=r.l1_decision,
                l2_score=r.l2_score,
                tier=r.tier,
                shap_top5=r.shap_top5 or [],
                analyst_action=r.analyst_action,
                model_version=r.model_version,
                prev_hash=r.prev_hash,
                curr_hash=r.curr_hash,
            )
            for r in rows
        ],
    )


@router.get("/events/recent", dependencies=[Depends(require_jwt)])
def recent_events(runtime: KavachRuntime = Depends(get_runtime)) -> JSONResponse:
    REQUEST_COUNT.labels(route="/events/recent", method="GET").inc()
    events = runtime.recent_events()
    return JSONResponse({"events": events})


@router.websocket("/transactions/feed")
async def transaction_feed(websocket: WebSocket) -> None:
    await require_ws_jwt(websocket)
    runtime: KavachRuntime = websocket.app.state.runtime
    await runtime.ws_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        await runtime.ws_hub.disconnect(websocket)

