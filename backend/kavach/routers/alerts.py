from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from kavach.dependencies import get_runtime
from kavach.runtime import KavachRuntime, REQUEST_COUNT
from kavach.schemas import AlertActionIn, AlertItem, RuleActionResponse
from kavach.security import require_jwt
from kavach.storage import AlertRecord, AuditLogRecord, session_scope


router = APIRouter(prefix="/alerts", tags=["alerts"], dependencies=[Depends(require_jwt)])


@router.get("/queue", response_model=list[AlertItem])
def get_alert_queue(
    limit: int = Query(default=200, ge=1, le=1000),
    runtime: KavachRuntime = Depends(get_runtime),
) -> list[AlertItem]:
    REQUEST_COUNT.labels(route="/alerts/queue", method="GET").inc()
    with session_scope() as session:
        rows = (
            session.query(AlertRecord)
            .filter(AlertRecord.status == "OPEN", AlertRecord.tier.in_(["High", "Critical"]))
            .order_by(AlertRecord.score.desc(), AlertRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            AlertItem(
                id=r.id,
                transaction_id=r.transaction_id,
                amount=r.amount,
                score=r.score,
                tier=r.tier,
                sender_state=r.sender_state,
                merchant_category=r.merchant_category,
                l1_decision=r.l1_decision,
                case_brief=r.case_brief,
                shap_top5=r.shap_top5 or [],
                l3_reconstruction_error=(
                    session.query(AuditLogRecord.l3_reconstruction_error)
                    .filter(AuditLogRecord.transaction_id == r.transaction_id)
                    .order_by(AuditLogRecord.id.desc())
                    .scalar()
                ),
                l3_graph_anomaly_score=(
                    session.query(AuditLogRecord.l3_graph_anomaly_score)
                    .filter(AuditLogRecord.transaction_id == r.transaction_id)
                    .order_by(AuditLogRecord.id.desc())
                    .scalar()
                ),
                created_at=r.created_at,
            )
            for r in rows
        ]


@router.post("/{alert_id}/action", response_model=RuleActionResponse)
async def submit_alert_action(
    alert_id: int,
    body: AlertActionIn,
    runtime: KavachRuntime = Depends(get_runtime),
) -> RuleActionResponse:
    REQUEST_COUNT.labels(route="/alerts/{id}/action", method="POST").inc()
    with session_scope() as session:
        row = session.get(AlertRecord, alert_id)
        if not row:
            raise HTTPException(status_code=404, detail="Alert not found")
        row.status = "RESOLVED"
        row.analyst_action = body.action.value
        row.analyst_note = body.analyst_note
        row.label = body.label
        row.updated_at = datetime.now(timezone.utc)

        audit = session.query(AuditLogRecord).filter(AuditLogRecord.transaction_id == row.transaction_id).order_by(AuditLogRecord.id.desc()).first()
        if audit:
            audit.analyst_action = body.action.value

    if body.label is not None:
        feats = {"txn_count_1m": 0.0, "amount_sum_1h": row.amount}
        runtime.drift_manager.update_labeled(feats, int(body.label), row.score)
        online = runtime.drift_manager.metrics()
        runtime.metrics.rolling_precision = online["precision"]
        runtime.metrics.rolling_recall = online["recall"]
        runtime.metrics.rolling_f2 = online["f2"]
        runtime.metrics.rolling_fpr = online["fpr"]
        runtime.metrics.drift_status = online["drift_status"]
        if online["drift_status"] == "retrain_urgent":
            await runtime.retrain_queue.put({"reason": "recall_drop_below_0.80"})

    return RuleActionResponse(ok=True, message=f"Alert {alert_id} marked {body.action.value}")
