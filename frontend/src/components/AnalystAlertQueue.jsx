import React from "react";

const tierClass = {
  High: "tier-high",
  Critical: "tier-critical"
};

export default function AnalystAlertQueue({ alerts, onAction }) {
  return (
    <div className="panel">
      <div className="panel-title">Analyst Alert Queue</div>
      <div className="alert-list">
        {alerts.map((a) => (
          <div key={a.id} className="alert-item">
            <div className="alert-main">
              <div><b>{a.transaction_id}</b></div>
              <div>INR {Number(a.amount || 0).toLocaleString("en-IN")}</div>
              <div className={`tier-badge ${tierClass[a.tier] || "tier-medium"}`}>{a.tier}</div>
              <div>Score {Number(a.score || 0).toFixed(3)}</div>
              <div>{a.sender_state}</div>
              <div>{new Date(a.created_at).toLocaleString()}</div>
            </div>
            <details>
              <summary>Case Brief</summary>
              <p>{a.case_brief || "No brief available."}</p>
              <p>
                L3: reconstruction {a.l3_reconstruction_error == null ? "-" : Number(a.l3_reconstruction_error).toFixed(4)}
                {" | "}graph anomaly {a.l3_graph_anomaly_score == null ? "-" : Number(a.l3_graph_anomaly_score).toFixed(4)}
              </p>
            </details>
            <div className="action-row">
              <button onClick={() => onAction(a.id, "Accept", 1)}>Accept</button>
              <button onClick={() => onAction(a.id, "Reject", 0)}>Reject</button>
              <button onClick={() => onAction(a.id, "Escalate", null)}>Escalate</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
