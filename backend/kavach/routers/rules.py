from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from kavach.dependencies import get_runtime
from kavach.runtime import KavachRuntime, REQUEST_COUNT
from kavach.schemas import PendingRuleItem, RuleActionResponse
from kavach.security import require_jwt
from kavach.storage import PendingRule, session_scope


router = APIRouter(prefix="/rules", tags=["rules"], dependencies=[Depends(require_jwt)])


@router.get("/pending", response_model=list[PendingRuleItem])
def pending_rules(runtime: KavachRuntime = Depends(get_runtime)) -> list[PendingRuleItem]:
    REQUEST_COUNT.labels(route="/rules/pending", method="GET").inc()
    with session_scope() as session:
        rows = session.query(PendingRule).filter(PendingRule.status == "PENDING").order_by(PendingRule.created_at.desc()).all()
        items = [
            PendingRuleItem(
                id=r.id,
                rule_name=r.rule_name,
                condition=r.condition,
                expected_recall_gain=r.expected_recall_gain,
                risk_of_fp=r.risk_of_fp,
                status=r.status,
                created_at=r.created_at,
            )
            for r in rows
        ]
    return items


@router.post("/{rule_id}/approve", response_model=RuleActionResponse)
def approve_rule(rule_id: int, runtime: KavachRuntime = Depends(get_runtime)) -> RuleActionResponse:
    REQUEST_COUNT.labels(route="/rules/{id}/approve", method="POST").inc()
    with session_scope() as session:
        row = session.get(PendingRule, rule_id)
        if not row:
            raise HTTPException(status_code=404, detail="Rule not found")
        rule_name = row.rule_name
        condition = row.condition.lower()
        row.status = "APPROVED"
        row.reviewed_at = datetime.now(timezone.utc)

    # Hot-load rule into L1 engine. For demo this parses bank-pair style patterns if present.
    if "sender_bank" in condition and "receiver_bank" in condition:
        # Analysts can provide explicit bank pair in rule_name as sender|receiver.
        if "|" in rule_name:
            sender, receiver = rule_name.split("|", 1)
            runtime.l1.add_block_rule(sender.strip(), receiver.strip())
    runtime.l1.blocklist_bank_pairs.add(f"RULE::{rule_name}")
    return RuleActionResponse(ok=True, message=f"Rule {rule_id} approved and hot-loaded")


@router.post("/{rule_id}/reject", response_model=RuleActionResponse)
def reject_rule(rule_id: int) -> RuleActionResponse:
    REQUEST_COUNT.labels(route="/rules/{id}/reject", method="POST").inc()
    with session_scope() as session:
        row = session.get(PendingRule, rule_id)
        if not row:
            raise HTTPException(status_code=404, detail="Rule not found")
        row.status = "REJECTED"
        row.reviewed_at = datetime.now(timezone.utc)
    return RuleActionResponse(ok=True, message=f"Rule {rule_id} rejected")
