# Kavach.ai Prompt Checklist

Status legend: `[x] done`, `[~] partial`, `[ ] pending`

## 1) Project Identity and Stack
- [x] Project naming and branding as `Kavach.ai` across backend/UI/docs
- [x] FastAPI backend + React frontend
- [x] Docker Compose with required services

## 2) Dataset 1 Pipeline (L1 + L2)
- [x] Temporal 70/15/15 split
- [x] Null handling, deduplication, outlier capping (99.9 percentile)
- [x] Categorical encoding (label/target by cardinality)
- [x] Imbalance pipeline: SMOTE -> ADASYN -> Tomek Links
- [x] Class weighting for XGBoost/LightGBM
- [x] Stratified K-Fold CV
- [x] Export held-out test split to `/exports/kavach_dataset1_test_split.csv`
- [x] Print split class distributions at startup

## 3) Dataset 2 Pipeline (L3)
- [x] Column narrowing for behavioral modeling set
- [x] Drop text/PII-heavy columns from modeling path
- [x] Median imputation + MinMax scaling
- [x] Stratified 80/20 split
- [x] Export held-out test split to `/exports/kavach_dataset2_test_split.csv`
- [x] Print split class distributions at startup

## 4) Three-Layer Detection
- [x] L1 rule gate with PASS/BLOCK/ESCALATE
- [x] L1 amount/velocity/weekend/sub-second checks
- [x] L1 low-threshold XGBoost recall gate
- [x] L2 ensemble (XGB + LGBM + IsolationForest + logistic stacker)
- [x] L2 calibration (Platt + isotonic)
- [x] L2 SHAP on medium+ scores
- [x] L3 async post-authorization enrichment
- [x] L3 LSTM autoencoder with 95th percentile anomaly threshold
- [x] L3 GraphSAGE training + embedding export + runtime anomaly integration

## 5) Feature Engineering / Feature Store
- [x] Dedicated feature module `kavach/features.py`
- [x] Redis-backed precomputed feature snapshots
- [x] Simulated Flink async feature stream updater
- [x] Inference path reads feature snapshot from Redis (no inline aggregation)

## 6) Defense Techniques
### 6.1 Graduated Tiered Response
- [x] 4-tier risk framework (Low/Medium/High/Critical) with mapped actions

### 6.2 Adversarial Hardening
- [x] FGSM-style adversarial augmentation (`epsilon=0.01`) with differentiable surrogate gradient
- [x] DP training via Opacus (target epsilon 0.5) for L3 autoencoder
- [x] Calibration audit flagging at inference
- [x] SHAP stability gating before promotion

### 6.3 Drift + Online Learning
- [x] River online learner (Hoeffding Tree + online logistic regression)
- [x] ADWIN drift detection + retrain queue trigger
- [x] Nightly retrain scheduling with APScheduler
- [x] MLflow tracking + candidate creation
- [x] Shadow-mode challenger scoring on identical live requests
- [x] Gated champion promotion (F2/AUC-PR/FPR + SHAP + shadow comparison checks)

### 6.4 API / Infra Defense
- [x] Redis sliding-window rate limiting
- [x] >10 req/min auto-escalation to High tier
- [x] JWT auth on all routes except `/health`
- [x] Request-space outlier flagging in audit (`request_outlier_flag`)
- [x] Circuit breaker with fallback to L1-only and auto-recovery

### 6.5 Audit / Non-Repudiation
- [x] Append-only audit table
- [x] SHA-256 hash chain verification
- [x] Training provenance ingestion into Postgres

### 6.6 Privacy
- [x] PII-sensitive fields excluded from feature store and audit usage
- [x] LLM prompts use structured, non-PII risk signals

## 7) AI Agents (n8n + Flowise)
- [x] Alert Narration Agent client integration
- [x] Tier >= High triggers narrated case brief
- [x] Adaptive Rule Agent trigger and pending-rules workflow
- [x] Rule approve/reject + hot-load in L1
- [x] Threat Intel Agent polling + Redis blocklist enrichment
- [x] Threat intel update persistence (source + timestamp)
- [x] n8n workflow definitions added (`n8n/workflows/*.json`) and auto-import wired in Docker Compose

## 8) FastAPI Endpoints
- [x] `/score`
- [x] `/audit-log`
- [x] `/metrics/live`
- [x] `/metrics/model`
- [x] `/alerts/queue`
- [x] `/alerts/{id}/action`
- [x] `/transactions/feed` (WebSocket)
- [x] `/exports/dataset1-test`
- [x] `/exports/dataset2-test`
- [x] `/rules/pending`
- [x] `/rules/{id}/approve`
- [x] `/rules/{id}/reject`
- [x] `/health`

## 9) React Dashboard Panels
- [x] KPI strip
- [x] Live transaction feed (WebSocket)
- [x] Risk score distribution histogram + tier thresholds
- [x] India fraud hotspot choropleth with tooltip
- [x] Analyst alert queue + actions + LLM brief + L3 context
- [x] Model performance panel + drift + champion/challenger + shadow comparison
- [x] Pending rules panel
- [x] Test split export panel

## 10) Simulator / Observability / Deployment
- [x] Async transaction simulator with configurable TPS
- [x] Fraud scenario injection (SIM swap, velocity attack, mule chain, overnight)
- [x] Prometheus metrics instrumentation
- [x] Grafana + Prometheus provisioning
- [x] Docker Compose health checks and service wiring
- [x] README with run/access/demo/export guidance

## 11) Final Validation
- [x] Python compile validation for backend modules
- [~] Full runtime validation (`docker compose up`, workflow connectivity, UI build in clean env) depends on local dependency install/runtime execution

## 12) Prompt Source
- [x] User-provided build prompt saved at `/prompt-to-do.txt`
