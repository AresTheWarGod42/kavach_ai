import React from "react";

export default function PendingRulesPanel({ rules, onApprove, onReject }) {
  return (
    <div className="panel">
      <div className="panel-title">Pending Rules</div>
      <table className="mini-table">
        <thead>
          <tr>
            <th>Rule</th>
            <th>Condition</th>
            <th>Recall Gain</th>
            <th>FP Risk</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {rules.map((r) => (
            <tr key={r.id}>
              <td>{r.rule_name}</td>
              <td>{r.condition}</td>
              <td>{(r.expected_recall_gain * 100).toFixed(2)}%</td>
              <td>{(r.risk_of_fp * 100).toFixed(2)}%</td>
              <td>
                <button onClick={() => onApprove(r.id)}>Approve</button>
                <button onClick={() => onReject(r.id)}>Reject</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

