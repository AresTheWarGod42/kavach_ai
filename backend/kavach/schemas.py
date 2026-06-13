from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class L1Decision(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


class RiskTier(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AnalystAction(str, Enum):
    ACCEPT = "Accept"
    REJECT = "Reject"
    ESCALATE = "Escalate"


class ShapContribution(BaseModel):
    feature: str
    feature_value: float | int | str | None = None
    shap_value: float


class TransactionIn(BaseModel):
    transaction_id: str
    timestamp: datetime
    transaction_type: str
    merchant_category: str
    amount: float = Field(ge=0.0)
    transaction_status: str
    sender_age_group: str
    receiver_age_group: str
    sender_state: str
    sender_bank: str
    receiver_bank: str
    device_type: str
    network_type: str
    user_id: str | None = None
    merchant_id: str | None = None
    receiver_account_age: float | None = None
    receiver_transaction_history: float | None = None
    transaction_amount_vs_sender_history: float | None = None
    geographic_disparity: float | None = None
    transaction_time_of_day: float | None = None
    merchant_category_code: float | None = None
    time_between_link_click_and_transaction: float | None = None
    unusual_device_flag: float | None = None
    unusual_ip_flag: float | None = None
    unusual_location_flag: float | None = None
    input_timing_consistency: float | None = None
    app_switching_frequency: float | None = None
    keyboard_input_speed: float | None = None
    screen_active_time: float | None = None
    geographic_location_vs_ip: float | None = None
    authentication_attempt_count: float | None = None
    time_between_otp_generation_and_input: float | None = None
    pin_entry_speed: float | None = None
    unusual_transaction_amount_flag: float | None = None
    otp_request_frequency: float | None = None
    transaction_velocity: float | None = None
    failed_transaction_count: float | None = None
    request_amount_roundness: float | None = None
    time_pressure_indicators: float | None = None
    handle_similarity_score: float | None = None
    handle_contains_official_terms: float | None = None
    handle_typo_analysis: float | None = None
    handle_verification_status: float | None = None


class ScoreResponse(BaseModel):
    project_name: str = "Kavach.ai"
    transaction_id: str
    l1_decision: L1Decision
    score: float
    confidence_interval: list[float]
    tier: RiskTier
    action: str
    circuit_breaker_fallback: bool
    shap: list[ShapContribution] = Field(default_factory=list)
    l3: dict[str, float | None] = Field(default_factory=dict)
    model_version: str
    calibration_audit_flag: bool = False
    request_outlier_flag: bool = False
    timestamp: datetime


class AlertItem(BaseModel):
    id: int
    transaction_id: str
    amount: float
    score: float
    tier: RiskTier
    sender_state: str
    merchant_category: str
    l1_decision: L1Decision
    case_brief: str | None = None
    shap_top5: list[dict[str, Any]] = Field(default_factory=list)
    l3_reconstruction_error: float | None = None
    l3_graph_anomaly_score: float | None = None
    created_at: datetime


class AlertActionIn(BaseModel):
    action: AnalystAction
    analyst_note: str | None = None
    label: int | None = Field(default=None, description="Fraud label for online learner: 1 fraud, 0 non-fraud")


class PendingRuleItem(BaseModel):
    id: int
    rule_name: str
    condition: str
    expected_recall_gain: float
    risk_of_fp: float
    status: str
    created_at: datetime


class LiveMetrics(BaseModel):
    tps: float
    p99_latency_ms: float
    fraud_rate_1m: float
    fraud_rate_5m: float
    fraud_rate_1h: float
    active_alerts: int
    model_version: str
    circuit_breaker_status: str


class ModelMetrics(BaseModel):
    precision: float
    recall: float
    f2: float
    auc_pr: float
    fpr: float
    drift_status: str
    champion_version: str
    challenger_version: str | None = None
    shadow_mode: bool = False
    shadow_samples: int = 0
    shadow_disagreement_rate: float = 0.0
    shadow_mean_abs_delta: float = 0.0
    shadow_champion_mean_score: float = 0.0
    shadow_challenger_mean_score: float = 0.0


class RuleActionResponse(BaseModel):
    ok: bool
    message: str


class AuditRecordOut(BaseModel):
    id: int
    transaction_id: str
    timestamp: datetime
    l1_decision: str
    l2_score: float | None
    tier: str | None
    shap_top5: list[dict[str, Any]]
    analyst_action: str | None
    model_version: str
    prev_hash: str | None
    curr_hash: str


class AuditLogResponse(BaseModel):
    total: int
    chain_valid: bool
    records: list[AuditRecordOut]
