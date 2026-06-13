from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc

from kavach.storage import AuditLogRecord, session_scope


def _hash_payload(payload: dict[str, Any], prev_hash: str | None) -> str:
    serial = json.dumps(payload, sort_keys=True, default=str)
    base = f"{prev_hash or 'GENESIS'}::{serial}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


class AuditLogger:
    def append(
        self,
        *,
        transaction_id: str,
        l1_decision: str,
        l2_score: float | None,
        tier: str | None,
        shap_top5: list[dict[str, Any]],
        analyst_action: str | None,
        model_version: str,
        l3_reconstruction_error: float | None = None,
        l3_graph_anomaly_score: float | None = None,
        calibration_audit_flag: bool = False,
        request_outlier_flag: bool = False,
    ) -> int:
        payload = {
            "transaction_id": transaction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "l1_decision": l1_decision,
            "l2_score": l2_score,
            "tier": tier,
            "shap_top5": shap_top5,
            "analyst_action": analyst_action,
            "model_version": model_version,
            "l3_reconstruction_error": l3_reconstruction_error,
            "l3_graph_anomaly_score": l3_graph_anomaly_score,
            "calibration_audit_flag": int(calibration_audit_flag),
            "request_outlier_flag": int(request_outlier_flag),
        }
        with session_scope() as session:
            last = session.query(AuditLogRecord).order_by(desc(AuditLogRecord.id)).first()
            prev_hash = last.curr_hash if last else None
            curr_hash = _hash_payload(payload, prev_hash)
            row = AuditLogRecord(
                transaction_id=transaction_id,
                timestamp=datetime.now(timezone.utc),
                l1_decision=l1_decision,
                l2_score=l2_score,
                tier=tier,
                shap_top5=shap_top5,
                analyst_action=analyst_action,
                model_version=model_version,
                prev_hash=prev_hash,
                curr_hash=curr_hash,
                l3_reconstruction_error=l3_reconstruction_error,
                l3_graph_anomaly_score=l3_graph_anomaly_score,
                calibration_audit_flag=int(calibration_audit_flag),
                request_outlier_flag=int(request_outlier_flag),
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def verify_chain(self, limit: int | None = None) -> tuple[bool, list[AuditLogRecord]]:
        with session_scope() as session:
            query = session.query(AuditLogRecord).order_by(AuditLogRecord.id.asc())
            if limit:
                rows = query.limit(limit).all()
            else:
                rows = query.all()
            prev_hash = None
            for row in rows:
                payload = {
                    "transaction_id": row.transaction_id,
                    "timestamp": row.timestamp.isoformat(),
                    "l1_decision": row.l1_decision,
                    "l2_score": row.l2_score,
                    "tier": row.tier,
                    "shap_top5": row.shap_top5,
                    "analyst_action": row.analyst_action,
                    "model_version": row.model_version,
                    "l3_reconstruction_error": row.l3_reconstruction_error,
                    "l3_graph_anomaly_score": row.l3_graph_anomaly_score,
                    "calibration_audit_flag": row.calibration_audit_flag,
                    "request_outlier_flag": row.request_outlier_flag,
                }
                expected = _hash_payload(payload, prev_hash)
                if row.curr_hash != expected or row.prev_hash != prev_hash:
                    return False, rows
                prev_hash = row.curr_hash
            return True, rows

