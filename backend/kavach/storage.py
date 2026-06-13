from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from kavach.config import settings


class Base(DeclarativeBase):
    pass


class AuditLogRecord(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    l1_decision: Mapped[str] = mapped_column(String(16))
    l2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    shap_top5: Mapped[list] = mapped_column(JSON, default=list)
    analyst_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str] = mapped_column(String(128))
    prev_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    curr_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    l3_reconstruction_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    l3_graph_anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_audit_flag: Mapped[int] = mapped_column(Integer, default=0)
    request_outlier_flag: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AlertRecord(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(128), index=True, unique=True)
    amount: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    tier: Mapped[str] = mapped_column(String(16), index=True)
    sender_state: Mapped[str] = mapped_column(String(64))
    merchant_category: Mapped[str] = mapped_column(String(128))
    l1_decision: Mapped[str] = mapped_column(String(16))
    case_brief: Mapped[str | None] = mapped_column(Text, nullable=True)
    shap_top5: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="OPEN", index=True)
    analyst_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    analyst_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class PendingRule(Base):
    __tablename__ = "pending_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_name: Mapped[str] = mapped_column(String(128))
    condition: Mapped[str] = mapped_column(Text)
    expected_recall_gain: Mapped[float] = mapped_column(Float)
    risk_of_fp: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    source_event: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelMetricsSnapshot(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f2: Mapped[float] = mapped_column(Float)
    auc_pr: Mapped[float] = mapped_column(Float)
    fpr: Mapped[float] = mapped_column(Float)
    drift_status: Mapped[str] = mapped_column(String(16), default="stable")
    champion_version: Mapped[str] = mapped_column(String(128))
    challenger_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    shadow_mode: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TrainingProvenance(Base):
    __tablename__ = "training_provenance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_dataset: Mapped[str] = mapped_column(String(256))
    ingest_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    label_origin: Mapped[str] = mapped_column(String(128))
    synthetic_flag: Mapped[int] = mapped_column(Integer, default=0)
    synthetic_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str] = mapped_column(String(128))


class ThreatIntelUpdate(Base):
    __tablename__ = "threat_intel_updates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sender_bank: Mapped[str] = mapped_column(String(128), index=True)
    receiver_bank: Mapped[str] = mapped_column(String(128), index=True)
    source: Mapped[str] = mapped_column(String(256))
    source_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


engine = create_engine(settings.postgres_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def count_open_alerts() -> int:
    with session_scope() as session:
        return int(
            session.query(func.count(AlertRecord.id))
            .filter(AlertRecord.status == "OPEN", AlertRecord.tier.in_(["High", "Critical"]))
            .scalar()
            or 0
        )
