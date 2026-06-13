from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Kavach.ai"
    app_tagline: str = "AI-Powered Real-Time UPI Fraud Shield"
    environment: str = "dev"

    jwt_secret: str = "kavach-dev-secret-change-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "kavach-ai"
    token_exp_minutes: int = 1440

    redis_url: str = "redis://redis:6379/0"
    postgres_url: str = "postgresql+psycopg://postgres:postgres@postgres:5432/kavach"
    mlflow_tracking_uri: str = "http://mlflow:5000"

    flowise_url: str = "http://flowise:3000"
    flowise_prediction_path: str = "/api/v1/prediction/kavach-alert"
    n8n_url: str = "http://n8n:5678"
    threat_intel_interval_hours: int = 6
    threat_intel_webhook_path: str = "/webhook/kavach-threat-intel"
    threat_intel_sources_csv: str = ""

    score_threshold_medium: float = 0.45
    score_threshold_high: float = 0.75
    score_threshold_critical: float = 0.96
    score_tolerance_gamma: float = 1.35
    l1_xgb_threshold: float = 0.30
    l1_velocity_escalate_threshold: int = 12
    l1_subsecond_min_interval_sec: float = 0.20

    l1_amount_ceiling_inr: float = 200000.0
    weekend_night_amount_inr: float = 10000.0
    overnight_large_transfer_inr: float = 150000.0

    throttle_limit_per_minute: int = 10
    throttle_force_tier: str = "High"

    circuit_breaker_trip_p99_ms: float = 200.0
    circuit_breaker_recover_p99_ms: float = 140.0

    simulator_enabled: bool = True
    simulator_tps: int = 50
    simulator_fraud_injection_rate: float = 0.03
    simulator_base_url: str = "http://localhost:8000"

    exports_dir: Path = Path("/app/backend/exports")
    models_dir: Path = Path("/app/backend/kavach/models")
    data_dir: Path = Path("/app/backend/data")

    dataset1_filename: str = "dataset_1.csv"
    dataset2_filename: str = "dataset_3.csv"
    dataset1_export_filename: str = "kavach_dataset1_test_split.csv"
    dataset2_export_filename: str = "kavach_dataset2_test_split.csv"

    model_version: str = "kavach-champion-v1"

    @property
    def dataset1_path(self) -> Path:
        return self.data_dir / self.dataset1_filename

    @property
    def dataset2_path(self) -> Path:
        return self.data_dir / self.dataset2_filename

    @property
    def dataset1_export_path(self) -> Path:
        return self.exports_dir / self.dataset1_export_filename

    @property
    def dataset2_export_path(self) -> Path:
        return self.exports_dir / self.dataset2_export_filename


settings = Settings()
