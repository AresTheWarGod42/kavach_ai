import React, { useEffect, useRef, useState } from "react";
import KPIStrip from "./components/KPIStrip";
import LiveTransactionFeed from "./components/LiveTransactionFeed";
import RiskScoreDistribution from "./components/RiskScoreDistribution";
import FraudHotspotMap from "./components/FraudHotspotMap";
import AnalystAlertQueue from "./components/AnalystAlertQueue";
import ModelPerformancePanel from "./components/ModelPerformancePanel";
import PendingRulesPanel from "./components/PendingRulesPanel";
import TestDataExport from "./components/TestDataExport";
import { apiGet, apiPost } from "./services/api";
import { openFeedSocket } from "./services/ws";

export default function App() {
  const [live, setLive] = useState({});
  const [modelMetrics, setModelMetrics] = useState({});
  const [distribution, setDistribution] = useState([]);
  const [hotspots, setHotspots] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [rules, setRules] = useState([]);
  const [events, setEvents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [tpsHistory, setTpsHistory] = useState([]);
  const [feedPaused, setFeedPaused] = useState(false);
  const feedPausedRef = useRef(false);
  const lastRenderAtRef = useRef(0);

  useEffect(() => {
    feedPausedRef.current = feedPaused;
  }, [feedPaused]);

  const refreshFast = async () => {
    try {
      const [liveRes, alertsRes] = await Promise.all([apiGet("/metrics/live"), apiGet("/alerts/queue?limit=200")]);
      setLive(liveRes.data);
      setAlerts(alertsRes.data || []);
      setTpsHistory((prev) => [...prev.slice(-59), { tps: liveRes.data?.tps || 0 }]);
    } catch (_) {}
  };

  const refreshMedium = async () => {
    const [distRes, rulesRes] = await Promise.all([apiGet("/metrics/distribution"), apiGet("/rules/pending")]);
    setDistribution(distRes.data?.histogram || []);
    setRules(rulesRes.data || []);
  };

  const refreshSlow = async () => {
    const hotspotRes = await apiGet("/metrics/hotspots");
    setHotspots(hotspotRes.data?.states || {});
  };

  const refreshModelHourly = async () => {
    const modelRes = await apiGet("/metrics/model");
    setModelMetrics(modelRes.data || {});
  };

  useEffect(() => {
    let ws = null;
    let disposed = false;
    refreshFast();
    refreshMedium();
    refreshSlow();
    refreshModelHourly();
    const fast = setInterval(refreshFast, 5000);
    const medium = setInterval(refreshMedium, 60000);
    const slow = setInterval(refreshSlow, 30000);
    const modelHourly = setInterval(refreshModelHourly, 3600000);
    openFeedSocket(
      (evt) => {
        if (disposed || feedPausedRef.current || !evt?.transaction_id) return;
        const now = Date.now();
        if (now - lastRenderAtRef.current < 200) return;
        lastRenderAtRef.current = now;
        setEvents((prev) => [evt, ...prev.slice(0, 199)]);
      },
      () => {}
    ).then((sock) => {
      ws = sock;
    });
    return () => {
      disposed = true;
      clearInterval(fast);
      clearInterval(medium);
      clearInterval(slow);
      clearInterval(modelHourly);
      if (ws) ws.close();
    };
  }, []);

  const onAlertAction = async (id, action, label) => {
    await apiPost(`/alerts/${id}/action`, { action, label, analyst_note: `Actioned via dashboard: ${action}` });
    await refreshFast();
  };

  const onApproveRule = async (id) => {
    await apiPost(`/rules/${id}/approve`, {});
    refreshMedium();
  };

  const onRejectRule = async (id) => {
    await apiPost(`/rules/${id}/reject`, {});
    refreshMedium();
  };

  return (
    <div className="app-shell">
      <header className="top-header">
        <div>
          <h1>Kavach.ai</h1>
          <p>AI-Powered Real-Time UPI Fraud Shield</p>
        </div>
      </header>
      <KPIStrip live={live} tpsHistory={tpsHistory} />
      <div className="grid">
        <div className="col-8">
          <LiveTransactionFeed
            rows={events}
            selectedId={selected?.transaction_id}
            onSelect={setSelected}
            paused={feedPaused}
            onTogglePaused={() => setFeedPaused((prev) => !prev)}
          />
          <RiskScoreDistribution histogram={distribution} />
          <FraudHotspotMap hotspots={hotspots} />
        </div>
        <div className="col-4">
          <AnalystAlertQueue alerts={alerts} onAction={onAlertAction} />
          <ModelPerformancePanel metrics={modelMetrics} />
          <PendingRulesPanel rules={rules} onApprove={onApproveRule} onReject={onRejectRule} />
          <TestDataExport />
          <div className="panel">
            <div className="panel-title">Transaction Detail</div>
            {selected ? (
              <>
                <div><b>ID:</b> {selected.transaction_id}</div>
                <div><b>Tier:</b> {selected.tier}</div>
                <div><b>L2 score:</b> {Number(selected.score || 0).toFixed(3)}</div>
                <div className="shap-list">
                  {(selected.shap || []).map((s, i) => (
                    <div key={i} className="shap-item">
                      <span>{s.feature}</span>
                      <span>{Number(s.shap_value || 0).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div>Select a feed row for SHAP breakdown.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
