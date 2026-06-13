import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export default function RiskScoreDistribution({ histogram }) {
  const spike = histogram?.find((x) => x.bucket >= 0.8)?.count || 0;
  return (
    <div className="panel">
      <div className="panel-title">Risk Score Distribution {spike > 20 ? <span className="alert-text">High critical-bin spike</span> : null}</div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={histogram || []}>
            <XAxis dataKey="bucket" />
            <YAxis />
            <Tooltip />
            <ReferenceLine x={0.3} stroke="#f6ad55" strokeDasharray="3 3" />
            <ReferenceLine x={0.6} stroke="#ed8936" strokeDasharray="3 3" />
            <ReferenceLine x={0.85} stroke="#f56565" strokeDasharray="3 3" />
            <Bar dataKey="count" fill="#2dd4bf" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

