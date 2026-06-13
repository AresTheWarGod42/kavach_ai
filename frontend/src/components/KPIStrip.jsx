import React from "react";
import { LineChart, Line, ResponsiveContainer } from "recharts";

export default function KPIStrip({ live, tpsHistory }) {
  const p99Hot = (live?.p99_latency_ms || 0) > 150;
  const circuitBad = live?.circuit_breaker_status === "L1_ONLY";
  return (
    <div className="panel kpi-strip">
      <div className="kpi-card">
        <div className="kpi-label">TPS</div>
        <div className="kpi-value">{(live?.tps || 0).toFixed(2)}</div>
        <div className="sparkline">
          <ResponsiveContainer width="100%" height={36}>
            <LineChart data={tpsHistory}>
              <Line type="monotone" dataKey="tps" dot={false} stroke="#1dd1a1" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">P99 Latency</div>
        <div className={`kpi-value ${p99Hot ? "hot" : ""}`}>{(live?.p99_latency_ms || 0).toFixed(1)} ms</div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Fraud Rate</div>
        <div className="kpi-badges">
          <span className="badge">{((live?.fraud_rate_1m || 0) * 100).toFixed(2)}% /1m</span>
          <span className="badge">{((live?.fraud_rate_5m || 0) * 100).toFixed(2)}% /5m</span>
          <span className="badge">{((live?.fraud_rate_1h || 0) * 100).toFixed(2)}% /1h</span>
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Active Alerts</div>
        <div className="kpi-value">{live?.active_alerts || 0}</div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Model Version</div>
        <div className="kpi-value small">{live?.model_version || "-"}</div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Circuit Breaker</div>
        <div className={`kpi-value ${circuitBad ? "hot" : "ok"}`}>{circuitBad ? "L1-only fallback" : "L1+L2 active"}</div>
      </div>
    </div>
  );
}

