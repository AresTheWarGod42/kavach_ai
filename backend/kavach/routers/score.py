from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from kavach.config import settings
from kavach.dependencies import get_runtime
from kavach.runtime import (
    CIRCUIT_BREAKER_GAUGE,
    FRAUD_COUNTER,
    LAYER_INVOCATIONS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    KavachRuntime,
    tier_from_score,
)
from kavach.schemas import RiskTier, ScoreResponse, TransactionIn
from kavach.security import require_jwt
from kavach.storage import AlertRecord, AuditLogRecord, session_scope


router = APIRouter(prefix="", tags=["score"], dependencies=[Depends(require_jwt)])


async def _run_l3_and_append(runtime: KavachRuntime, txn: dict, audit_id: int) -> None:
    result = await runtime.l3.enrich(txn)
    rec_error = float(result.get("reconstruction_error", 0.0))
    graph_score = float(result.get("graph_anomaly_score", 0.0))
    alert_snapshot: dict | None = None
    with session_scope() as session:
        audit = session.get(AuditLogRecord, audit_id)
        if audit:
            audit.l3_reconstruction_error = rec_error
            audit.l3_graph_anomaly_score = graph_score
        alert = session.query(AlertRecord).filter(AlertRecord.transaction_id == txn.get("transaction_id")).first()
        if alert:
            alert_snapshot = {
                "alert_id": int(alert.id),
                "score": float(alert.score),
                "tier": str(alert.tier),
                "amount": float(alert.amount),
                "sender_state": str(alert.sender_state),
                "merchant_category": str(alert.merchant_category),
                "shap_top5": list(alert.shap_top5 or []),
            }

    if alert_snapshot is not None:
        case_payload = {
            "transaction_id": txn.get("transaction_id"),
            "score": alert_snapshot["score"],
            "tier": alert_snapshot["tier"],
            "shap_top5": alert_snapshot["shap_top5"],
            "amount": alert_snapshot["amount"],
            "sender_state": alert_snapshot["sender_state"],
            "merchant_category": alert_snapshot["merchant_category"],
            "hour_of_day": int(datetime.now(timezone.utc).hour),
            "l3_reconstruction_error": rec_error,
            "l3_graph_anomaly_score": graph_score,
        }
        brief = await runtime.narration_agent.create_case_brief(case_payload)
        with session_scope() as session:
            row = session.get(AlertRecord, alert_snapshot["alert_id"])
            if row:
                row.case_brief = brief

    if rec_error >= runtime.l3.threshold or graph_score >= 0.8:
        await runtime.retrain_queue.put(
            {
                "reason": "l3_behavioral_anomaly_cluster",
                "transaction_id": txn.get("transaction_id"),
                "reconstruction_error": rec_error,
                "graph_anomaly_score": graph_score,
            }
        )


@router.post("/score", response_model=ScoreResponse)
async def score_transaction(
    payload: TransactionIn,
    runtime: KavachRuntime = Depends(get_runtime),
) -> ScoreResponse:
    REQUEST_COUNT.labels(route="/score", method="POST").inc()
    started = time.perf_counter()
    transaction = payload.model_dump()
    if payload.timestamp.tzinfo is None:
        transaction["timestamp"] = payload.timestamp.replace(tzinfo=timezone.utc)

    user_id = str(transaction.get("user_id") or transaction["transaction_id"])
    rate = runtime.rate_limiter.check(user_id)

    # Inference path only reads precomputed features from Redis.
    cached_features = runtime.feature_store.read_snapshot(transaction)
    await runtime.feature_stream_queue.put(transaction)

    LAYER_INVOCATIONS.labels(layer="l1").inc()
    l1 = runtime.l1.evaluate(transaction, cached_features)

    use_l2 = l1.decision == "ESCALATE" and not runtime.circuit_breaker.fallback_mode
    if use_l2:
        LAYER_INVOCATIONS.labels(layer="l2").inc()
        l2_out = runtime.l2.score(transaction, cached_features)
        score = float(l2_out["score"])
        ci = [float(x) for x in l2_out["confidence_interval"]]
        shap = l2_out["shap"]
        calibration_audit_flag = bool(l2_out["calibration_audit_flag"])
        request_outlier_flag = bool(l2_out["request_outlier_flag"])

        if runtime.metrics.shadow_mode and runtime.shadow_l2 is not None:
            try:
                shadow_out = runtime.shadow_l2.score(transaction, cached_features)
                shadow_score = float(shadow_out["score"])
                champion_tier, _ = tier_from_score(score)
                challenger_tier, _ = tier_from_score(shadow_score)
                runtime.metrics.track_shadow(score, shadow_score, champion_tier, challenger_tier)
            except Exception:
                pass
    else:
        fallback_score = 0.9 if l1.decision == "BLOCK" else 0.15
        score = fallback_score
        ci = [max(0.0, fallback_score - 0.1), min(1.0, fallback_score + 0.1)]
        shap = []
        calibration_audit_flag = False
        request_outlier_flag = False

    tier_name, action = tier_from_score(score)
    if rate.force_high_tier and tier_name in {"Low", "Medium"}:
        tier_name = "High"
        action = "Step-up auth trigger due to rate-limit escalation"

    now_dt = datetime.now(timezone.utc)
    is_fraud = tier_name in {"High", "Critical"} or l1.decision == "BLOCK"
    if is_fraud:
        FRAUD_COUNTER.labels(tier=tier_name).inc()

    if tier_name in {"High", "Critical"}:
        async with runtime.alert_lock:
            case_payload = {
                "transaction_id": transaction["transaction_id"],
                "score": score,
                "tier": tier_name,
                "shap_top5": shap,
                "amount": float(transaction.get("amount", 0.0)),
                "sender_state": transaction.get("sender_state"),
                "merchant_category": transaction.get("merchant_category"),
                "hour_of_day": int(now_dt.hour),
            }
            brief = await runtime.narration_agent.create_case_brief(case_payload)
            with session_scope() as session:
                existing = session.query(AlertRecord).filter(AlertRecord.transaction_id == transaction["transaction_id"]).first()
                if not existing:
                    alert = AlertRecord(
                        transaction_id=transaction["transaction_id"],
                        amount=float(transaction.get("amount", 0.0)),
                        score=score,
                        tier=tier_name,
                        sender_state=str(transaction.get("sender_state", "Unknown")),
                        merchant_category=str(transaction.get("merchant_category", "Unknown")),
                        l1_decision=l1.decision,
                        shap_top5=shap[:5],
                        case_brief=brief,
                        status="OPEN",
                    )
                    session.add(alert)

    audit_id = runtime.audit_logger.append(
        transaction_id=transaction["transaction_id"],
        l1_decision=l1.decision,
        l2_score=score if use_l2 else None,
        tier=tier_name,
        shap_top5=shap[:5],
        analyst_action=None,
        model_version=runtime.model_version,
        calibration_audit_flag=calibration_audit_flag,
        request_outlier_flag=request_outlier_flag,
    )

    l3_payload: dict[str, float] = {}
    if l1.decision == "ESCALATE":
        LAYER_INVOCATIONS.labels(layer="l3").inc()
        asyncio.create_task(_run_l3_and_append(runtime, transaction, audit_id))
        l3_payload = {"queued": 1.0}

    drift = runtime.drift_manager.update_unlabeled_fraud_rate(score)
    runtime.metrics.drift_status = runtime.drift_manager.drift_state
    if drift.drift_detected:
        await runtime.retrain_queue.put({"reason": drift.reason, "ts": now_dt.isoformat()})

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    runtime.metrics.track(elapsed_ms, score, str(transaction.get("sender_state", "Unknown")), is_fraud)
    runtime.circuit_breaker.record(elapsed_ms)
    CIRCUIT_BREAKER_GAUGE.set(1 if runtime.circuit_breaker.fallback_mode else 0)
    runtime.push_event(
        {
            "transaction_id": transaction["transaction_id"],
            "amount": float(transaction.get("amount", 0.0)),
            "sender_state": str(transaction.get("sender_state", "Unknown")),
            "merchant_category": str(transaction.get("merchant_category", "Unknown")),
            "device_type": str(transaction.get("device_type", "Unknown")),
            "hour": now_dt.hour,
            "l1_decision": l1.decision,
            "score": score,
            "tier": tier_name,
            "timestamp": now_dt.isoformat(),
            "shap": shap,
        }
    )
    await runtime.ws_hub.broadcast(runtime.metrics.scored_events[0])
    REQUEST_LATENCY.labels(route="/score").observe(elapsed_ms / 1000.0)

    return ScoreResponse(
        transaction_id=transaction["transaction_id"],
        l1_decision=l1.decision,
        score=score,
        confidence_interval=ci,
        tier=RiskTier(tier_name),
        action=action,
        circuit_breaker_fallback=runtime.circuit_breaker.fallback_mode,
        shap=shap,
        l3=l3_payload,
        model_version=runtime.model_version,
        calibration_audit_flag=calibration_audit_flag,
        request_outlier_flag=request_outlier_flag,
        timestamp=now_dt,
    )
