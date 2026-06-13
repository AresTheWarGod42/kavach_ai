import React from "react";

function driftClass(status) {
  if (status === "stable") return "ok";
  if (status === "drift_detected") return "warn";
  return "hot";
}

export default function ModelPerformancePanel({ metrics }) {
  return (
    <div className="panel">
      <div className="panel-title">Model Performance</div>
      <div className="metric-grid">
        <div className="metric-card"><span>Precision</span><b>{(metrics?.precision || 0).toFixed(3)}</b></div>
        <div className="metric-card"><span>Recall</span><b>{(metrics?.recall || 0).toFixed(3)}</b></div>
        <div className="metric-card"><span>F2</span><b>{(metrics?.f2 || 0).toFixed(3)}</b></div>
        <div className="metric-card"><span>AUC-PR</span><b>{(metrics?.auc_pr || 0).toFixed(3)}</b></div>
        <div className="metric-card"><span>FPR</span><b>{((metrics?.fpr || 0) * 100).toFixed(3)}%</b></div>
      </div>
      <div className="drift-row">
        Drift:
        <span className={`status-dot ${driftClass(metrics?.drift_status)}`}></span>
        <span>{metrics?.drift_status || "stable"}</span>
      </div>
      <table className="mini-table">
        <thead>
          <tr>
            <th>Mode</th>
            <th>Version</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Champion</td>
            <td>{metrics?.champion_version || "-"}</td>
          </tr>
          <tr>
            <td>Challenger</td>
            <td>{metrics?.challenger_version || "-"}</td>
          </tr>
        </tbody>
      </table>
      {metrics?.shadow_mode ? (
        <table className="mini-table" style={{ marginTop: "8px" }}>
          <thead>
            <tr>
              <th>Shadow Comparison</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Samples</td>
              <td>{metrics?.shadow_samples || 0}</td>
            </tr>
            <tr>
              <td>Disagreement Rate</td>
              <td>{((metrics?.shadow_disagreement_rate || 0) * 100).toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Mean |Δ score|</td>
              <td>{(metrics?.shadow_mean_abs_delta || 0).toFixed(4)}</td>
            </tr>
            <tr>
              <td>Champion Avg Score</td>
              <td>{(metrics?.shadow_champion_mean_score || 0).toFixed(4)}</td>
            </tr>
            <tr>
              <td>Challenger Avg Score</td>
              <td>{(metrics?.shadow_challenger_mean_score || 0).toFixed(4)}</td>
            </tr>
          </tbody>
        </table>
      ) : null}
    </div>
  );
}
