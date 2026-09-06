import React from 'react';
import { TrendingUp } from 'lucide-react';

export default function IntensityChart({ historicalPoints = [], forecastPoints = [] }) {
  // Combine historical and forecast points
  const points = [
    ...historicalPoints.map((p, i) => ({
      label: p.timestamp ? p.timestamp.split(' ')[1] || `T-${historicalPoints.length - 1 - i}` : `T-${historicalPoints.length - 1 - i}`,
      vmax: p.vmax,
      isForecast: false
    })),
    ...forecastPoints.map(p => ({
      label: `+${p.horizon}`,
      vmax: p.predicted_vmax_kt,
      isForecast: true
    }))
  ];

  const maxVmax = Math.max(140, ...points.map(p => p.vmax + 15));
  const minVmax = 20;

  const width = 500;
  const height = 180;
  const paddingLeft = 45;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 30;

  const chartW = width - paddingLeft - paddingRight;
  const chartH = height - paddingTop - paddingBottom;

  const getX = (index) => {
    if (points.length <= 1) return paddingLeft;
    return paddingLeft + (index / (points.length - 1)) * chartW;
  };

  const getY = (val) => {
    return paddingTop + chartH - ((val - minVmax) / (maxVmax - minVmax)) * chartH;
  };

  // Build SVG path
  let pathD = "";
  let forecastPathD = "";

  const histCount = historicalPoints.length;

  points.forEach((p, idx) => {
    const x = getX(idx);
    const y = getY(p.vmax);

    if (idx === 0) {
      pathD += `M ${x} ${y}`;
    } else if (idx < histCount) {
      pathD += ` L ${x} ${y}`;
    }

    if (idx === histCount - 1) {
      forecastPathD += `M ${x} ${y}`;
    } else if (idx >= histCount) {
      forecastPathD += ` L ${x} ${y}`;
    }
  });

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <TrendingUp size={18} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600 }}>Intensity Trend & Category Thresholds</h3>
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          Knots (Vmax) | Synoptic Timeline
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Threshold horizontal guides */}
          {/* Very Strong (83 kt) */}
          <line
            x1={paddingLeft}
            y1={getY(83)}
            x2={width - paddingRight}
            y2={getY(83)}
            stroke="rgba(244, 63, 94, 0.35)"
            strokeDasharray="3, 3"
          />
          <text x={paddingLeft + 4} y={getY(83) - 4} fill="#f43f5e" fontSize="9" fontWeight="600">
            Very Strong (≥83 kt)
          </text>

          {/* Strong (64 kt) */}
          <line
            x1={paddingLeft}
            y1={getY(64)}
            x2={width - paddingRight}
            y2={getY(64)}
            stroke="rgba(249, 115, 22, 0.35)"
            strokeDasharray="3, 3"
          />
          <text x={paddingLeft + 4} y={getY(64) - 4} fill="#f97316" fontSize="9" fontWeight="600">
            Strong (64-82 kt)
          </text>

          {/* Moderate (34 kt) */}
          <line
            x1={paddingLeft}
            y1={getY(34)}
            x2={width - paddingRight}
            y2={getY(34)}
            stroke="rgba(245, 158, 11, 0.35)"
            strokeDasharray="3, 3"
          />
          <text x={paddingLeft + 4} y={getY(34) - 4} fill="#f59e0b" fontSize="9" fontWeight="600">
            Moderate (34-63 kt)
          </text>

          {/* Y Axis Grid Labels */}
          {[30, 60, 90, 120].map((v) => (
            <text key={`lbl-${v}`} x={paddingLeft - 8} y={getY(v) + 3} fill="#64748b" fontSize="9" textAnchor="end">
              {v}k
            </text>
          ))}

          {/* Historical Line */}
          {pathD && (
            <path
              d={pathD}
              fill="none"
              stroke="#38bdf8"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
          )}

          {/* Forecast Line (Dashed) */}
          {forecastPathD && (
            <path
              d={forecastPathD}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="2.5"
              strokeDasharray="4, 4"
              strokeLinecap="round"
            />
          )}

          {/* Points */}
          {points.map((p, idx) => {
            const x = getX(idx);
            const y = getY(p.vmax);
            const isFc = p.isForecast;

            return (
              <g key={`pt-${idx}`}>
                <circle
                  cx={x}
                  cy={y}
                  r={isFc ? 4.5 : 3.5}
                  fill={isFc ? "#f59e0b" : "#38bdf8"}
                  stroke="#080c14"
                  strokeWidth="2"
                />
                <text
                  x={x}
                  y={y - 8}
                  fill={isFc ? "#fbbf24" : "#f8fafc"}
                  fontSize="9"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  {Math.round(p.vmax)}
                </text>
                {/* X Axis Label */}
                <text
                  x={x}
                  y={height - 8}
                  fill={isFc ? "#f59e0b" : "#94a3b8"}
                  fontSize="9"
                  textAnchor="middle"
                >
                  {p.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
