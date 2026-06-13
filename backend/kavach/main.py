from __future__ import annotations

import asyncio
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from redis import Redis

from kavach.agents.narration import NarrationAgent
from kavach.agents.rule_agent import AdaptiveRuleAgent
from kavach.agents.threat_intel import ThreatIntelAgent
from kavach.config import settings
from kavach.defense.audit_log import AuditLogger
from kavach.defense.circuit_breaker import LatencyCircuitBreaker
from kavach.defense.drift_detector import OnlineDriftManager
from kavach.defense.rate_limiter import SlidingWindowRateLimiter
from kavach.features import FeatureComputer, seed_features_from_dataset, simulated_flink_feature_job
from kavach.layers.l1_gate import L1Gate
from kavach.layers.l2_ensemble import L2Ensemble
from kavach.layers.l3_deep import L3DeepDetector
from kavach.runtime import KavachRuntime
from kavach.routers import alerts, exports, metrics, rules, score
from kavach.security import create_access_token
from kavach.storage import PendingRule, ThreatIntelUpdate, TrainingProvenance, init_db, session_scope
from kavach.simulator import TransactionSimulator
from training.train_l1_l2 import prepare_dataset1_splits_and_exports, train_and_export_l1_l2
from training.train_l3 import prepare_dataset2_splits_and_exports, train_and_export_l3


def _ensure_dataset_files() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    source_1 = root / "data" / settings.dataset1_filename
    source_2 = root / "data" / settings.dataset2_filename
    if not settings.dataset1_path.exists() and source_1.exists():
        shutil.copy2(source_1, settings.dataset1_path)
    if not settings.dataset2_path.exists() and source_2.exists():
        shutil.copy2(source_2, settings.dataset2_path)


def _print_split_stats() -> None:
    split1 = prepare_dataset1_splits_and_exports(settings.dataset1_path, settings.dataset1_export_path)
    print("[Kavach.ai] Dataset1 train class distribution:", split1["stats"]["train"])
    print("[Kavach.ai] Dataset1 val class distribution:", split1["stats"]["val"])
    print("[Kavach.ai] Dataset1 test class distribution:", split1["stats"]["test"])
    split2 = prepare_dataset2_splits_and_exports(settings.dataset2_path, settings.dataset2_export_path)
    print("[Kavach.ai] Dataset2 train class distribution:", split2["stats"]["train"])
    print("[Kavach.ai] Dataset2 test class distribution:", split2["stats"]["test"])


def _ingest_provenance_if_exists() -> None:
    for file_name in ["provenance_dataset1.csv", "provenance_dataset2.csv"]:
        path = settings.models_dir / file_name
        if not path.exists():
            continue
        import pandas as pd

        df = pd.read_csv(path)
        with session_scope() as session:
            for row in df.to_dict(orient="records"):
                exists = (
                    session.query(TrainingProvenance)
                    .filter(
                        TrainingProvenance.source_dataset == str(row.get("source_dataset")),
                        TrainingProvenance.synthetic_flag == int(row.get("synthetic_flag", 0)),
                        TrainingProvenance.model_version == str(row.get("model_version")),
                    )
                    .first()
                )
                if exists:
                    continue
                session.add(
                    TrainingProvenance(
                        source_dataset=str(row.get("source_dataset")),
                        ingest_timestamp=datetime.fromisoformat(str(row.get("ingest_timestamp")).replace("Z", "+00:00")),
                        label_origin=str(row.get("label_origin")),
                        synthetic_flag=int(row.get("synthetic_flag", 0)),
                        synthetic_method=str(row.get("synthetic_method")) if row.get("synthetic_method") else None,
                        model_version=str(row.get("model_version", settings.model_version)),
                    )
                )


async def _drift_retrain_worker(runtime: KavachRuntime) -> None:
    while not runtime.stop_event.is_set():
        event = await runtime.retrain_queue.get()
        try:
            reason = event.get("reason", "drift_event")
            cluster = {"recent_high_tier": random.randint(15, 90), "reason": reason}
            proposals = await runtime.rule_agent.propose_rules(reason, cluster)
            with session_scope() as session:
                for p in proposals[:3]:
                    row = PendingRule(
                        rule_name=p["rule_name"],
                        condition=p["condition"],
                        expected_recall_gain=float(p["expected_recall_gain"]),
                        risk_of_fp=float(p["risk_of_fp"]),
                        status="PENDING",
                        source_event=reason,
                    )
                    session.add(row)
        finally:
            runtime.retrain_queue.task_done()


async def _threat_intel_worker(runtime: KavachRuntime) -> None:
    while not runtime.stop_event.is_set():
        updates = await runtime.threat_agent.poll()
        for item in updates:
            runtime.l1.add_block_rule(item["sender_bank"], item["receiver_bank"])
            with session_scope() as session:
                exists = (
                    session.query(ThreatIntelUpdate)
                    .filter(
                        ThreatIntelUpdate.sender_bank == str(item["sender_bank"]),
                        ThreatIntelUpdate.receiver_bank == str(item["receiver_bank"]),
                        ThreatIntelUpdate.source == str(item.get("source", "unknown")),
                    )
                    .first()
                )
                if exists:
                    continue
                raw_ts = str(item.get("timestamp") or datetime.now(timezone.utc).isoformat())
                session.add(
                    ThreatIntelUpdate(
                        sender_bank=str(item["sender_bank"]),
                        receiver_bank=str(item["receiver_bank"]),
                        source=str(item.get("source", "unknown")),
                        source_timestamp=datetime.fromisoformat(raw_ts.replace("Z", "+00:00")),
                    )
                )
        await asyncio.sleep(settings.threat_intel_interval_hours * 3600)


def _candidate_shap_top5(candidate_dir: Path) -> list[str]:
    meta = candidate_dir / "l2_metadata.json"
    if not meta.exists():
        return []
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(x) for x in payload.get("shap_baseline_top5", [])]


def _promote_candidate(candidate_dir: Path, live_dir: Path) -> None:
    for file in candidate_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, live_dir / file.name)


def _activate_shadow_candidate(
    runtime: KavachRuntime,
    *,
    challenger: str,
    candidate_dir: Path,
    metrics_l1_l2: dict[str, float],
    shap_ok: bool,
) -> None:
    shadow_model = L2Ensemble(models_dir=candidate_dir)
    shadow_model.load()
    runtime.shadow_l2 = shadow_model
    runtime.shadow_candidate_dir = str(candidate_dir)
    runtime.shadow_candidate_metrics = {k: float(v) for k, v in metrics_l1_l2.items()}
    runtime.shadow_shap_ok = bool(shap_ok)
    runtime.metrics.challenger_version = challenger
    runtime.metrics.shadow_mode = True
    runtime.metrics.reset_shadow_tracking()
    print("[Kavach.ai] Activated challenger in shadow mode:", challenger)


def _maybe_promote_existing_shadow(runtime: KavachRuntime) -> bool:
    if not runtime.metrics.shadow_mode or runtime.shadow_l2 is None or not runtime.shadow_candidate_dir:
        return False

    samples_ok = runtime.metrics.shadow_samples >= 200
    disagreement_ok = runtime.metrics.shadow_disagreement_rate <= 0.30
    gate_ok = (
        float(runtime.shadow_candidate_metrics.get("f2", 0.0)) > 0.87
        and float(runtime.shadow_candidate_metrics.get("auc_pr", 0.0)) > 0.92
        and float(runtime.shadow_candidate_metrics.get("fpr", 1.0)) < 0.005
        and runtime.shadow_shap_ok
        and samples_ok
        and disagreement_ok
    )
    if not gate_ok:
        print(
            "[Kavach.ai] Shadow challenger not promoted yet:",
            {
                "samples": runtime.metrics.shadow_samples,
                "disagreement_rate": round(runtime.metrics.shadow_disagreement_rate, 4),
                "f2": runtime.shadow_candidate_metrics.get("f2"),
                "auc_pr": runtime.shadow_candidate_metrics.get("auc_pr"),
                "fpr": runtime.shadow_candidate_metrics.get("fpr"),
                "shap_ok": runtime.shadow_shap_ok,
            },
        )
        return True

    candidate_dir = Path(runtime.shadow_candidate_dir)
    challenger = runtime.metrics.challenger_version or f"{runtime.model_version}-shadow"
    _promote_candidate(candidate_dir, settings.models_dir)
    runtime.l1.load()
    runtime.l2.load()
    runtime.l3.load()
    runtime.model_version = challenger
    runtime.shadow_l2 = None
    runtime.shadow_candidate_dir = None
    runtime.shadow_candidate_metrics = {}
    runtime.shadow_shap_ok = True
    runtime.metrics.shadow_mode = False
    runtime.metrics.challenger_version = None
    runtime.metrics.reset_shadow_tracking()
    print("[Kavach.ai] Promoted challenger to champion after shadow evaluation:", challenger)
    return True


def _nightly_retrain_job(runtime: KavachRuntime) -> None:
    # If a challenger is already shadow-running, evaluate whether to promote.
    if _maybe_promote_existing_shadow(runtime):
        return

    challenger = f"{runtime.model_version}-shadow-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    candidate_dir = settings.models_dir / "candidates" / challenger
    candidate_dir.mkdir(parents=True, exist_ok=True)

    prev_model_version = settings.model_version
    settings.model_version = challenger
    try:
        metrics_l1_l2 = train_and_export_l1_l2(settings.dataset1_path, candidate_dir, settings.dataset1_export_path)
        _ = train_and_export_l3(settings.dataset2_path, candidate_dir, settings.dataset2_export_path)
    except Exception as exc:
        print("[Kavach.ai] Nightly retrain failed:", exc)
        settings.model_version = prev_model_version
        return
    finally:
        settings.model_version = prev_model_version

    runtime.metrics.rolling_precision = float(metrics_l1_l2.get("precision", runtime.metrics.rolling_precision))
    runtime.metrics.rolling_recall = float(metrics_l1_l2.get("recall", runtime.metrics.rolling_recall))
    runtime.metrics.rolling_f2 = float(metrics_l1_l2.get("f2", runtime.metrics.rolling_f2))
    runtime.metrics.rolling_auc_pr = float(metrics_l1_l2.get("auc_pr", runtime.metrics.rolling_auc_pr))
    runtime.metrics.rolling_fpr = float(metrics_l1_l2.get("fpr", runtime.metrics.rolling_fpr))

    candidate_top5 = _candidate_shap_top5(candidate_dir)
    shap_ok = True
    if runtime.l2.shap_baseline_top5 and candidate_top5:
        overlap = len(set(runtime.l2.shap_baseline_top5).intersection(candidate_top5))
        shap_ok = overlap >= 2
    _activate_shadow_candidate(
        runtime,
        challenger=challenger,
        candidate_dir=candidate_dir,
        metrics_l1_l2=metrics_l1_l2,
        shap_ok=shap_ok,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Kavach.ai", version="1.0.0", description=settings.app_tagline)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/internal/prometheus", make_asgi_app())
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    app.state.scheduler = scheduler

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "project": "Kavach.ai",
                "time": datetime.now(timezone.utc).isoformat(),
                "demo_token": create_access_token("kavach-frontend", {"role": "analyst"}),
            }
        )

    app.include_router(score.router)
    app.include_router(alerts.router)
    app.include_router(metrics.router)
    app.include_router(exports.router)
    app.include_router(rules.router)

    @app.on_event("startup")
    async def startup() -> None:
        settings.exports_dir.mkdir(parents=True, exist_ok=True)
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        _ensure_dataset_files()
        _print_split_stats()
        init_db()
        _ingest_provenance_if_exists()

        redis_client = Redis.from_url(settings.redis_url, decode_responses=False)
        feature_store = FeatureComputer(redis_client)
        seed_features_from_dataset(str(settings.dataset1_path), feature_store, limit=10000)

        l1 = L1Gate(redis=redis_client, model_path=settings.models_dir / "l1_xgb.onnx")
        l1.load()
        l2 = L2Ensemble(models_dir=settings.models_dir)
        l2.load()
        l3 = L3DeepDetector(models_dir=settings.models_dir)
        l3.load()

        runtime = KavachRuntime(
            redis=redis_client,
            l1=l1,
            l2=l2,
            l3=l3,
            feature_store=feature_store,
            rate_limiter=SlidingWindowRateLimiter(redis_client),
            audit_logger=AuditLogger(),
            circuit_breaker=LatencyCircuitBreaker(),
            drift_manager=OnlineDriftManager(),
            narration_agent=NarrationAgent(),
            rule_agent=AdaptiveRuleAgent(),
            threat_agent=ThreatIntelAgent(),
        )
        runtime.generated_token = create_access_token("kavach-service", {"role": "internal"})
        app.state.runtime = runtime

        print("[Kavach.ai] JWT token for internal calls:", runtime.generated_token)

        scheduler.add_job(_nightly_retrain_job, "cron", hour=2, minute=0, args=[runtime], id="nightly-retrain")
        scheduler.start()

        app.state.feature_task = asyncio.create_task(
            simulated_flink_feature_job(runtime.feature_stream_queue, feature_store, runtime.stop_event)
        )
        app.state.drift_task = asyncio.create_task(_drift_retrain_worker(runtime))
        app.state.threat_task = asyncio.create_task(_threat_intel_worker(runtime))

        if settings.simulator_enabled:
            simulator = TransactionSimulator(
                dataset_path=str(settings.dataset1_path),
                token=runtime.generated_token,
                tps=settings.simulator_tps,
                fraud_injection_rate=settings.simulator_fraud_injection_rate,
                base_url=settings.simulator_base_url,
            )
            app.state.simulator = simulator
            app.state.sim_task = asyncio.create_task(simulator.run())

    @app.on_event("shutdown")
    async def shutdown() -> None:
        runtime: KavachRuntime = app.state.runtime
        runtime.stop_event.set()
        for key in ["feature_task", "drift_task", "threat_task", "sim_task"]:
            task = getattr(app.state, key, None)
            if task:
                task.cancel()
        scheduler.shutdown(wait=False)
        if runtime.redis:
            runtime.redis.close()

    return app


app = create_app()


def main() -> None:
    uvicorn.run("kavach.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
