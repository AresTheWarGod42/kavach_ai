import React from "react";

const tierClass = {
  Low: "tier-low",
  Medium: "tier-medium",
  High: "tier-high",
  Critical: "tier-critical"
};

export default function LiveTransactionFeed({ rows, selectedId, onSelect, paused, onTogglePaused }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <div className="panel-title">Live Transaction Feed</div>
        <button type="button" onClick={onTogglePaused}>
          {paused ? "Resume" : "Pause"}
        </button>
      </div>
      <div className="table-wrap">
        <table className="feed-table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Amount (INR)</th>
              <th>Sender State</th>
              <th>Merchant Category</th>
              <th>Device Type</th>
              <th>Hour</th>
              <th>L1 Decision</th>
              <th>L2 Score</th>
              <th>Tier</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 500).map((row) => (
              <tr
                key={row.transaction_id}
                className={selectedId === row.transaction_id ? "selected" : ""}
                onClick={() => onSelect(row)}
              >
                <td>{row.transaction_id}</td>
                <td>{Number(row.amount || 0).toLocaleString("en-IN")}</td>
                <td>{row.sender_state}</td>
                <td>{row.merchant_category}</td>
                <td>{row.device_type}</td>
                <td>{row.hour}</td>
                <td>{row.l1_decision}</td>
                <td>{Number(row.score || 0).toFixed(3)}</td>
                <td>
                  <span className={`tier-badge ${tierClass[row.tier]}`}>{row.tier}</span>
                </td>
                <td>{new Date(row.timestamp).toLocaleTimeString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
