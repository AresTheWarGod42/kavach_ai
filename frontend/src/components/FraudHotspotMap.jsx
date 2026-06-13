import React, { useMemo, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";

const geoUrl = "/india-states.geojson";

function colorFor(rate) {
  if (rate > 0.2) return "#f43f5e";
  if (rate > 0.1) return "#fb923c";
  if (rate > 0.05) return "#f59e0b";
  if (rate > 0.01) return "#84cc16";
  return "#1f2937";
}

export default function FraudHotspotMap({ hotspots }) {
  const [hover, setHover] = useState(null);
  const stateData = useMemo(() => hotspots || {}, [hotspots]);

  return (
    <div className="panel">
      <div className="panel-title">Fraud Hotspot by State</div>
      <div className="map-wrap">
        <ComposableMap projection="geoMercator" projectionConfig={{ center: [82, 22], scale: 900 }}>
          <Geographies geography={geoUrl}>
            {({ geographies }) =>
              geographies.map((geo) => {
                const name = geo.properties.st_nm || geo.properties.NAME_1 || geo.properties.name;
                const row = stateData[name] || { fraud_count: 0, total: 0, fraud_rate: 0 };
                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    onMouseEnter={() => setHover({ name, ...row })}
                    onMouseLeave={() => setHover(null)}
                    style={{
                      default: { fill: colorFor(row.fraud_rate), stroke: "#111827", strokeWidth: 0.8 },
                      hover: { fill: "#38bdf8", stroke: "#111827", strokeWidth: 0.9 },
                      pressed: { fill: "#38bdf8" }
                    }}
                  />
                );
              })
            }
          </Geographies>
        </ComposableMap>
        <div className="map-tooltip">
          {hover ? (
            <>
              <div>{hover.name}</div>
              <div>Fraud: {Math.round(hover.fraud_count)}</div>
              <div>Total: {Math.round(hover.total)}</div>
              <div>Rate: {(hover.fraud_rate * 100).toFixed(2)}%</div>
            </>
          ) : (
            <div>Hover a state</div>
          )}
        </div>
      </div>
    </div>
  );
}

