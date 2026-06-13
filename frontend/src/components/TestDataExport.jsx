import React from "react";
import { apiBase, getToken } from "../services/api";

async function download(path) {
  const token = await getToken();
  const res = await fetch(`${apiBase()}${path}`, { headers: { Authorization: `Bearer ${token}` } });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = path.includes("dataset1") ? "kavach_dataset1_test_split.csv" : "kavach_dataset2_test_split.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export default function TestDataExport() {
  return (
    <div className="panel">
      <div className="panel-title">Test Data Export</div>
      <div className="action-row">
        <button onClick={() => download("/exports/dataset1-test")}>Download Dataset 1 Test Split</button>
        <button onClick={() => download("/exports/dataset2-test")}>Download Dataset 2 Test Split</button>
      </div>
    </div>
  );
}

