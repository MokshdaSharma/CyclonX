import React, { useState, useEffect } from 'react';
import { 
  Compass, 
  Wind, 
  Gauge, 
  Layers, 
  AlertCircle, 
  RefreshCw, 
  Activity, 
  Radio,
  Satellite
} from 'lucide-react';

import MapTrackViewer from './components/MapTrackViewer';
import SatelliteViewer from './components/SatelliteViewer';
import IntensityChart from './components/IntensityChart';
import RiskAuditPanel from './components/RiskAuditPanel';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Built-in fallback demo dataset for offline zero-config operation
const DEFAULT_STORMS = {
  "FANI_2019": {
    name: "Extremely Severe Cyclonic Storm Fani (2019)",
    basin: "Bay of Bengal",
    current_vmax_kt: 115.0,
    category: "Very Strong",
    current_lat: 14.8,
    current_lon: 84.2,
    history: [
      { timestamp: "2019-04-29 00:00 UTC", lat: 8.5, lon: 87.0, vmax: 45.0, pres: 996.0 },
      { timestamp: "2019-04-29 12:00 UTC", lat: 9.2, lon: 86.8, vmax: 55.0, pres: 990.0 },
      { timestamp: "2019-04-30 00:00 UTC", lat: 10.3, lon: 86.2, vmax: 75.0, pres: 978.0 },
      { timestamp: "2019-04-30 12:00 UTC", lat: 11.6, lon: 85.5, vmax: 95.0, pres: 962.0 },
      { timestamp: "2019-05-01 00:00 UTC", lat: 13.1, lon: 84.8, vmax: 105.0, pres: 950.0 },
      { timestamp: "2019-05-01 12:00 UTC", lat: 14.8, lon: 84.2, vmax: 115.0, pres: 938.0 }
    ],
    forecast: [
      { horizon: "6h", lat: 16.1, lon: 84.5, predicted_vmax_kt: 120.0, uncertainty_radius_km: 38.0 },
      { horizon: "12h", lat: 17.5, lon: 85.1, predicted_vmax_kt: 125.0, uncertainty_radius_km: 68.0 },
      { horizon: "24h", lat: 19.8, lon: 85.8, predicted_vmax_kt: 115.0, uncertainty_radius_km: 125.0 }
    ],
    risk: {
      overall_risk_level: "Very High",
      total_risk_score: 95,
      intensity_threat: "Catastrophic (Category 4-5 Equivalent)",
      proximity_threat: "Immediate Coastal Threat (< 100 km)",
      rapid_intensification_risk: true,
      rule_audit_trace: [
        { rule_id: "RULE_INT_01", condition: "Current Vmax >= 83 kt (Very Strong)", triggered: true, risk_score_impact: 35, rationale: "Catastrophic storm surge and destructive core wind field." },
        { rule_id: "RULE_RI_01", condition: "Forecast 24h Vmax increase >= 20 kt", triggered: true, risk_score_impact: 20, rationale: "Rapid Intensification event detected." },
        { rule_id: "RULE_PROX_01", condition: "Distance to coast <= 100 km", triggered: true, risk_score_impact: 30, rationale: "Storm tracking directly towards Odisha coastline margin." },
        { rule_id: "RULE_UNCERT_01", condition: "24h Forecast Uncertainty Cone > 120 km", triggered: true, risk_score_impact: 10, rationale: "Broad emergency warning corridor required." }
      ]
    }
  },
  "AMPHAN_2020": {
    name: "Super Cyclonic Storm Amphan (2020)",
    basin: "Bay of Bengal",
    current_vmax_kt: 130.0,
    category: "Very Strong",
    current_lat: 14.2,
    current_lon: 86.3,
    history: [
      { timestamp: "2020-05-16 12:00 UTC", lat: 10.8, lon: 86.5, vmax: 40.0, pres: 998.0 },
      { timestamp: "2020-05-17 00:00 UTC", lat: 11.5, lon: 86.2, vmax: 55.0, pres: 988.0 },
      { timestamp: "2020-05-17 12:00 UTC", lat: 12.5, lon: 86.4, vmax: 75.0, pres: 972.0 },
      { timestamp: "2020-05-18 00:00 UTC", lat: 13.4, lon: 86.3, vmax: 115.0, pres: 935.0 },
      { timestamp: "2020-05-18 12:00 UTC", lat: 14.2, lon: 86.3, vmax: 130.0, pres: 920.0 }
    ],
    forecast: [
      { horizon: "6h", lat: 15.6, lon: 86.5, predicted_vmax_kt: 130.0, uncertainty_radius_km: 35.0 },
      { horizon: "12h", lat: 17.2, lon: 86.8, predicted_vmax_kt: 125.0, uncertainty_radius_km: 65.0 },
      { horizon: "24h", lat: 21.2, lon: 88.2, predicted_vmax_kt: 100.0, uncertainty_radius_km: 120.0 }
    ],
    risk: {
      overall_risk_level: "Very High",
      total_risk_score: 95,
      intensity_threat: "Catastrophic Super Cyclone",
      proximity_threat: "Sundarbans / West Bengal Coastal Corridor",
      rapid_intensification_risk: true,
      rule_audit_trace: [
        { rule_id: "RULE_INT_01", condition: "Current Vmax >= 83 kt", triggered: true, risk_score_impact: 35, rationale: "Super Cyclonic energy dissipation hazard." },
        { rule_id: "RULE_PROX_02", condition: "Coastline within 24h uncertainty radius", triggered: true, risk_score_impact: 20, rationale: "Direct impact corridor in West Bengal / Bangladesh." }
      ]
    }
  },
  "BIPARJOY_2023": {
    name: "Very Severe Cyclonic Storm Biparjoy (2023)",
    basin: "Arabian Sea",
    current_vmax_kt: 75.0,
    category: "Strong",
    current_lat: 17.2,
    current_lon: 67.2,
    history: [
      { timestamp: "2023-06-08 00:00 UTC", lat: 13.5, lon: 66.2, vmax: 50.0, pres: 990.0 },
      { timestamp: "2023-06-08 12:00 UTC", lat: 14.2, lon: 66.0, vmax: 65.0, pres: 980.0 },
      { timestamp: "2023-06-09 00:00 UTC", lat: 15.0, lon: 66.1, vmax: 75.0, pres: 974.0 },
      { timestamp: "2023-06-09 12:00 UTC", lat: 16.0, lon: 66.3, vmax: 80.0, pres: 970.0 },
      { timestamp: "2023-06-10 00:00 UTC", lat: 17.2, lon: 67.2, vmax: 75.0, pres: 975.0 }
    ],
    forecast: [
      { horizon: "6h", lat: 18.1, lon: 67.5, predicted_vmax_kt: 75.0, uncertainty_radius_km: 40.0 },
      { horizon: "12h", lat: 19.4, lon: 68.0, predicted_vmax_kt: 70.0, uncertainty_radius_km: 72.0 },
      { horizon: "24h", lat: 21.8, lon: 68.9, predicted_vmax_kt: 65.0, uncertainty_radius_km: 130.0 }
    ],
    risk: {
      overall_risk_level: "High",
      total_risk_score: 65,
      intensity_threat: "Severe (Very Severe Cyclonic Storm)",
      proximity_threat: "Gujarat / Saurashtra Coastline Approaching",
      rapid_intensification_risk: false,
      rule_audit_trace: [
        { rule_id: "RULE_INT_02", condition: "Current Vmax between 64 and 82 kt (Strong)", triggered: true, risk_score_impact: 25, rationale: "Severe hurricane-force winds." },
        { rule_id: "RULE_PROX_03", condition: "Distance to coast between 100 and 350 km", triggered: true, risk_score_impact: 20, rationale: "Tracking north-northeast towards Kutch/Saurashtra." },
        { rule_id: "RULE_UNCERT_01", condition: "24h Forecast Uncertainty Cone > 120 km", triggered: true, risk_score_impact: 10, rationale: "Track recurvature introduces spatial spread." }
      ]
    }
  },
  "MOCHA_2023": {
    name: "Extremely Severe Cyclonic Storm Mocha (2023)",
    basin: "Bay of Bengal",
    current_vmax_kt: 120.0,
    category: "Very Strong",
    current_lat: 16.2,
    current_lon: 90.0,
    history: [
      { timestamp: "2023-05-11 06:00 UTC", lat: 11.2, lon: 88.2, vmax: 45.0, pres: 994.0 },
      { timestamp: "2023-05-11 18:00 UTC", lat: 12.1, lon: 87.8, vmax: 60.0, pres: 984.0 },
      { timestamp: "2023-05-12 06:00 UTC", lat: 13.2, lon: 88.0, vmax: 80.0, pres: 968.0 },
      { timestamp: "2023-05-12 18:00 UTC", lat: 14.6, lon: 88.7, vmax: 105.0, pres: 948.0 },
      { timestamp: "2023-05-13 06:00 UTC", lat: 16.2, lon: 90.0, vmax: 120.0, pres: 932.0 }
    ],
    forecast: [
      { horizon: "6h", lat: 17.5, lon: 91.1, predicted_vmax_kt: 125.0, uncertainty_radius_km: 36.0 },
      { horizon: "12h", lat: 18.9, lon: 92.0, predicted_vmax_kt: 120.0, uncertainty_radius_km: 64.0 },
      { horizon: "24h", lat: 20.8, lon: 93.4, predicted_vmax_kt: 95.0, uncertainty_radius_km: 115.0 }
    ],
    risk: {
      overall_risk_level: "Very High",
      total_risk_score: 90,
      intensity_threat: "Catastrophic Extremely Severe CS",
      proximity_threat: "Myanmar / Bangladesh Coastline Corridor",
      rapid_intensification_risk: true,
      rule_audit_trace: [
        { rule_id: "RULE_INT_01", condition: "Current Vmax >= 83 kt", triggered: true, risk_score_impact: 35, rationale: "Extreme wind shear and surge danger." },
        { rule_id: "RULE_RI_01", condition: "Forecast 24h Vmax increase >= 20 kt", triggered: true, risk_score_impact: 20, rationale: "Rapid intensification verified." },
        { rule_id: "RULE_PROX_01", condition: "Distance to coast <= 100 km", triggered: true, risk_score_impact: 30, rationale: "Direct landfall impending on Sittwe / Cox's Bazar margin." }
      ]
    }
  }
};

export default function App() {
  const [selectedStormKey, setSelectedStormKey] = useState("FANI_2019");
  const [apiConnected, setApiConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeStormData, setActiveStormData] = useState(DEFAULT_STORMS["FANI_2019"]);
  const [gradcamBase64, setGradcamBase64] = useState(null);
  const [showCustomModal, setShowCustomModal] = useState(false);
  const [customForm, setCustomForm] = useState({
    name: "Custom Cyclone Sim",
    basin: "Bay of Bengal",
    lat: 15.5,
    lon: 86.2,
    vmax: 90.0,
    pres: 965.0,
    heading_dlat: 0.45,
    heading_dlon: -0.25
  });

  // Check API health on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'healthy') {
          setApiConnected(true);
        }
      })
      .catch(() => {
        setApiConnected(false);
      });
  }, []);

  // Update active storm data when selection changes
  const handleSelectStorm = async (key) => {
    setSelectedStormKey(key);
    const storm = DEFAULT_STORMS[key];
    setActiveStormData(storm);

    if (apiConnected) {
      setLoading(true);
      try {
        // Query /predict to obtain live model intensity and Grad-CAM overlay
        const predRes = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cyclone_id: key, return_gradcam: true, gradcam_alpha: 0.45 })
        });
        if (predRes.ok) {
          const predJson = await predRes.json();
          setGradcamBase64(predJson.gradcam_overlay_base64);
        }

        // Query /predict-track
        const trackRes = await fetch(`${API_BASE_URL}/predict-track`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cyclone_id: key, observations: storm.history })
        });
        if (trackRes.ok) {
          const trackJson = await trackRes.json();
          storm.forecast = trackJson.forecast_horizons;
        }

        // Query /predict-risk
        const riskRes = await fetch(`${API_BASE_URL}/predict-risk`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cyclone_id: key,
            current_vmax_kt: storm.current_vmax_kt,
            current_lat: storm.current_lat,
            current_lon: storm.current_lon
          })
        });
        if (riskRes.ok) {
          const riskJson = await riskRes.json();
          storm.risk = riskJson;
        }
      } catch (err) {
        console.warn("API query failed, continuing with built-in telemetry", err);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleRunCustomSimulation = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    const lat = parseFloat(customForm.lat) || 15.0;
    const lon = parseFloat(customForm.lon) || 86.0;
    const vmax = parseFloat(customForm.vmax) || 80.0;
    const pres = parseFloat(customForm.pres) || 970.0;
    const dlat = parseFloat(customForm.heading_dlat) || 0.45;
    const dlon = parseFloat(customForm.heading_dlon) || -0.25;

    const history = [
      { timestamp: "T -18h", lat: Math.round((lat - dlat * 3) * 10) / 10, lon: Math.round((lon - dlon * 3) * 10) / 10, vmax: Math.max(30, vmax - 20), pres: pres + 15 },
      { timestamp: "T -12h", lat: Math.round((lat - dlat * 2) * 10) / 10, lon: Math.round((lon - dlon * 2) * 10) / 10, vmax: Math.max(30, vmax - 12), pres: pres + 10 },
      { timestamp: "T -6h",  lat: Math.round((lat - dlat * 1) * 10) / 10, lon: Math.round((lon - dlon * 1) * 10) / 10, vmax: Math.max(30, vmax - 5), pres: pres + 4 },
      { timestamp: "Latest Observation", lat: lat, lon: lon, vmax: vmax, pres: pres }
    ];

    let forecast = [
      { horizon: "6h", lat: Math.round((lat + dlat * 1) * 100) / 100, lon: Math.round((lon + dlon * 1) * 100) / 100, predicted_vmax_kt: vmax + 3, uncertainty_radius_km: 38 },
      { horizon: "12h", lat: Math.round((lat + dlat * 2.2) * 100) / 100, lon: Math.round((lon + dlon * 2.1) * 100) / 100, predicted_vmax_kt: vmax + 5, uncertainty_radius_km: 68 },
      { horizon: "24h", lat: Math.round((lat + dlat * 4.5) * 100) / 100, lon: Math.round((lon + dlon * 4.2) * 100) / 100, predicted_vmax_kt: Math.max(35, vmax + 2), uncertainty_radius_km: 125 }
    ];

    const category = vmax >= 83 ? "Very Strong" : (vmax >= 64 ? "Strong" : (vmax >= 34 ? "Moderate" : "Weak"));
    let riskObj = null;

    if (apiConnected) {
      try {
        const trackRes = await fetch(`${API_BASE_URL}/predict-track`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cyclone_id: "CUSTOM_SIM", observations: history })
        });
        if (trackRes.ok) {
          const trackJson = await trackRes.json();
          forecast = trackJson.forecast_horizons;
        }

        const riskRes = await fetch(`${API_BASE_URL}/predict-risk`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cyclone_id: "CUSTOM_SIM",
            current_vmax_kt: vmax,
            current_lat: lat,
            current_lon: lon
          })
        });
        if (riskRes.ok) {
          riskObj = await riskRes.json();
        }
      } catch (err) {
        console.warn("API simulation fallback", err);
      }
    }

    const customObj = {
      name: customForm.name || "Custom Storm Simulation",
      basin: customForm.basin || "North Indian Ocean",
      current_vmax_kt: vmax,
      category: category,
      current_lat: lat,
      current_lon: lon,
      history: history,
      forecast: forecast,
      risk: riskObj || {
        overall_risk_level: vmax >= 83 ? "Very High" : (vmax >= 64 ? "High" : "Moderate"),
        total_risk_score: Math.min(100, Math.round(vmax * 0.75)),
        intensity_threat: `${category} Wind Hazard`,
        proximity_threat: "Simulated Coastal Perimeter",
        rapid_intensification_risk: vmax > 75,
        rule_audit_trace: [
          { rule_id: "RULE_CUSTOM_01", condition: `Simulated Vmax = ${vmax} kt`, triggered: true, risk_score_impact: Math.round(vmax * 0.35), rationale: "User parameterized simulation input." }
        ]
      }
    };

    setActiveStormData(customObj);
    setSelectedStormKey("CUSTOM");
    setShowCustomModal(false);
  };

  const currentStorm = activeStormData || DEFAULT_STORMS["FANI_2019"];
  const latestObs = currentStorm.history[currentStorm.history.length - 1];

  const getCategoryBadge = (cat) => {
    switch (cat.toLowerCase()) {
      case 'very strong': return 'badge-very-strong';
      case 'strong': return 'badge-strong';
      case 'moderate': return 'badge-moderate';
      default: return 'badge-weak';
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-logo">
            <Radio size={22} />
          </div>
          <div>
            <div className="brand-title">CycloneX</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              SIH Problem Statement 26070 | Ministry of Earth Sciences / IMD
            </div>
          </div>
        </div>

        {/* Controls & API Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button
            onClick={() => setShowCustomModal(true)}
            className="btn-primary"
            style={{ fontSize: '0.8rem', padding: '0.4rem 0.85rem' }}
          >
            <Compass size={14} /> Simulate Custom Storm
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
            <span style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: apiConnected ? 'var(--accent-emerald)' : '#64748b',
              boxShadow: apiConnected ? '0 0 8px var(--accent-emerald)' : 'none'
            }}></span>
            <span style={{ color: apiConnected ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              {apiConnected ? 'API Connected (v1.0.0)' : 'Client Fallback Mode'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Dashboard Container */}
      <main className="dashboard-container" style={{ flex: 1 }}>
        {/* Controls and Cyclone Switcher */}
        <div className="top-bar-controls">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Select Tropical Cyclone:
            </span>
            <div className="cyclone-selector-group">
              {Object.keys(DEFAULT_STORMS).map((key) => (
                <button
                  key={key}
                  onClick={() => handleSelectStorm(key)}
                  className={`btn-select-storm ${selectedStormKey === key ? 'active' : ''}`}
                >
                  {key.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span className={`badge ${getCategoryBadge(currentStorm.category)}`}>
              {currentStorm.category}
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Basin: <strong>{currentStorm.basin}</strong>
            </span>
          </div>
        </div>

        {/* Quick Stats Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
          <div className="stat-box glass-panel">
            <div className="stat-label">Estimated Current Intensity (Vmax)</div>
            <div className="stat-value" style={{ color: 'var(--accent-cyan)' }}>
              {currentStorm.current_vmax_kt} <span style={{ fontSize: '1rem', fontWeight: 400 }}>kt</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              ~{Math.round(currentStorm.current_vmax_kt * 1.852)} km/h sustained
            </div>
          </div>

          <div className="stat-box glass-panel">
            <div className="stat-label">Current Eye Coordinates</div>
            <div className="stat-value">
              {latestObs.lat}°N, {latestObs.lon}°E
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Observation: {latestObs.timestamp || 'Latest Synoptic Frame'}
            </div>
          </div>

          <div className="stat-box glass-panel">
            <div className="stat-label">Estimated Central Pressure</div>
            <div className="stat-value" style={{ color: 'var(--accent-amber)' }}>
              {latestObs.pres || 950} <span style={{ fontSize: '1rem', fontWeight: 400 }}>hPa</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Deep convective central core
            </div>
          </div>

          <div className="stat-box glass-panel">
            <div className="stat-label">24h Forecast Horizon</div>
            <div className="stat-value" style={{ color: 'var(--accent-rose)' }}>
              {currentStorm.forecast[2]?.predicted_vmax_kt || 110} <span style={{ fontSize: '1rem', fontWeight: 400 }}>kt</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Uncertainty Radius: ±{currentStorm.forecast[2]?.uncertainty_radius_km || 120} km
            </div>
          </div>
        </div>

        {/* Core Layout: Left Column (Map + Chart), Right Column (Satellite Viewer + Risk Panel) */}
        <div className="dashboard-grid">
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <MapTrackViewer
              historicalPoints={currentStorm.history}
              forecastPoints={currentStorm.forecast}
              cycloneName={currentStorm.name}
            />
            <IntensityChart
              historicalPoints={currentStorm.history}
              forecastPoints={currentStorm.forecast}
            />
          </div>

          {/* Right Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <SatelliteViewer
              gradcamBase64={gradcamBase64}
              cycloneId={selectedStormKey}
              predictedVmax={currentStorm.current_vmax_kt}
              intensityCategory={currentStorm.category}
            />
            <RiskAuditPanel
              riskData={currentStorm.risk}
            />
          </div>
        </div>
      </main>

      {/* Custom Storm Simulation Modal */}
      {showCustomModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 2000,
          padding: '1rem'
        }}>
          <div className="glass-panel" style={{
            maxWidth: '520px',
            width: '100%',
            padding: '1.75rem',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-highlight)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Compass size={20} color="var(--accent-cyan)" />
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Simulate Custom Cyclone</h3>
              </div>
              <button
                onClick={() => setShowCustomModal(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', fontSize: '1.2rem', cursor: 'pointer' }}
              >
                &times;
              </button>
            </div>

            <form onSubmit={handleRunCustomSimulation} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '0.75rem' }}>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Cyclone Name</label>
                  <input
                    type="text"
                    value={customForm.name}
                    onChange={(e) => setCustomForm({ ...customForm, name: e.target.value })}
                    style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: '#fff', padding: '0.45rem 0.65rem', borderRadius: '6px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Basin</label>
                  <select
                    value={customForm.basin}
                    onChange={(e) => setCustomForm({ ...customForm, basin: e.target.value })}
                    style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: '#fff', padding: '0.45rem 0.65rem', borderRadius: '6px' }}
                  >
                    <option value="Bay of Bengal">Bay of Bengal</option>
                    <option value="Arabian Sea">Arabian Sea</option>
                    <option value="Global Basin">Global Basin</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Latitude (°N)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={customForm.lat}
                    onChange={(e) => setCustomForm({ ...customForm, lat: e.target.value })}
                    style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: '#fff', padding: '0.45rem 0.65rem', borderRadius: '6px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Longitude (°E)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={customForm.lon}
                    onChange={(e) => setCustomForm({ ...customForm, lon: e.target.value })}
                    style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: '#fff', padding: '0.45rem 0.65rem', borderRadius: '6px' }}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Current Vmax (Knots)</label>
                  <input
                    type="number"
                    step="1"
                    value={customForm.vmax}
                    onChange={(e) => setCustomForm({ ...customForm, vmax: e.target.value })}
                    style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: '#fff', padding: '0.45rem 0.65rem', borderRadius: '6px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Central MSLP (hPa)</label>
                  <input
                    type="number"
                    step="1"
                    value={customForm.pres}
                    onChange={(e) => setCustomForm({ ...customForm, pres: e.target.value })}
                    style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: '#fff', padding: '0.45rem 0.65rem', borderRadius: '6px' }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
                <button
                  type="button"
                  onClick={() => setShowCustomModal(false)}
                  style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)', padding: '0.45rem 1rem', borderRadius: '6px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary"
                  style={{ padding: '0.45rem 1.25rem' }}
                >
                  Run AI Forecast & Risk Model
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="app-footer">
        <div>
          <strong>CycloneX</strong> &bull; AI-Powered Tropical Cyclone Decision Support Platform
        </div>
        <div>
          Model Engine: <code>cyclonex-v1.0.0</code>
        </div>
      </footer>
    </div>
  );
}
