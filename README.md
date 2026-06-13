# Kavach.ai

Kavach.ai is an AI-powered real-time UPI fraud defense system built for the NPCI UPI Fraud Detection Blue Team challenge.

## Stack

- Backend: FastAPI (`backend/kavach`)
- Frontend: React + Recharts + WebSockets (`frontend`)
- Feature store/rate limiting: Redis
- Audit log + provenance: PostgreSQL
- Tracking/registry: MLflow
- Monitoring: Prometheus + Grafana
- Agents: n8n + Flowise
- Scheduler: APScheduler (Airflow reference in `airflow/`)

## Run

```bash
docker compose up --build
```

## Service URLs

- Dashboard: http://localhost:3000
- FastAPI docs: http://localhost:8000/docs
- FastAPI health: http://localhost:8000/health
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)
- n8n: http://localhost:5678
- Flowise: http://localhost:3002

## Agent Workflows (n8n + Flowise)

- n8n workflow JSONs are provided in `n8n/workflows/` and are auto-imported on container startup.
- Webhook paths used by backend agents:
  - `POST /webhook/kavach-adaptive-rule`
  - `GET /webhook/kavach-threat-intel`
- Flowise narration prompt template is in `flowise/prompts/kavach-alert-system-prompt.txt`.
- Default backend Flowise prediction path is `/api/v1/prediction/kavach-alert`. If your Flowise chatflow id differs, set `FLOWISE_PREDICTION_PATH` for `kavach-api`.

## Training and exports

Two test-split files are exported to `backend/exports` and served by API:

- `kavach_dataset1_test_split.csv` via `GET /exports/dataset1-test`
- `kavach_dataset2_test_split.csv` via `GET /exports/dataset2-test`

The dashboard includes a **Test Data Export** panel with download buttons for both.

## Manual fraud scenario trigger

1. Start stack.
2. Open `http://localhost:3000`.
3. Use any API client with bearer token from `GET /health` field `demo_token`.
4. Call `POST /score` with a suspicious transaction:

```json
{
  "transaction_id": "MANUAL-FRAUD-1",
  "timestamp": "2026-03-30T02:47:10+05:30",
  "transaction_type": "P2M",
  "merchant_category": "Electronics",
  "amount": 185000,
  "transaction_status": "SUCCESS",
  "sender_age_group": "26-35",
  "receiver_age_group": "26-35",
  "sender_state": "Rajasthan",
  "sender_bank": "RISKYBANK_77",
  "receiver_bank": "RISKYBANK_88",
  "device_type": "NewAndroid",
  "network_type": "4G",
  "user_id": "manual-user-1",
  "merchant_id": "manual-merchant-1",
  "unusual_device_flag": 1,
  "unusual_location_flag": 1,
  "pin_entry_speed": 0.4,
  "time_between_otp_generation_and_input": 0.6
}
```

## Dashboard panels

1. **KPI Strip**: TPS, P99 latency, fraud-rate badges (1m/5m/1h), active alerts, model version, circuit breaker.
2. **Live Transaction Feed**: latest scored transactions with color-coded risk tiers and transaction details.
3. **Risk Score Distribution**: 5-minute histogram with 0.30/0.60/0.85 thresholds.
4. **Fraud Hotspot Map**: India state-level fraud heat intensity.
5. **Analyst Alert Queue**: High/Critical alerts with Flowise-generated case brief and analyst actions.
6. **Model Performance Panel**: Precision, Recall, F2, AUC-PR, FPR, drift status, champion/challenger.
7. **Pending Rules Panel**: adaptive rule proposals with approve/reject actions.
8. **Test Data Export Panel**: downloads both held-out test CSVs.

## Notes

- All API routes except `/health` require JWT bearer token.
- `/score` runs L1 -> L2 synchronously and launches L3 async enrichment.
- Prometheus scrape endpoint is mounted at `/internal/prometheus`.
- Dataset files are expected at `backend/data/dataset_1.csv` and `backend/data/dataset_3.csv`.
