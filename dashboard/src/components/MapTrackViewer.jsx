import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';

// Custom Map Recenter Helper
function ChangeView({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center && center.length === 2 && !isNaN(center[0]) && !isNaN(center[1])) {
      map.setView(center, zoom);
    }
  }, [center, zoom, map]);
  return null;
}

// Custom Leaflet DivIcons
const createDotIcon = (color, label, isPulse = false) => {
  return L.divIcon({
    className: 'custom-map-pin',
    html: `
      <div style="
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        ${isPulse ? `<div style="
          position: absolute;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: ${color};
          opacity: 0.4;
          animation: mapPulse 2s infinite ease-out;
        "></div>` : ''}
        <div style="
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: ${color};
          border: 2px solid #ffffff;
          box-shadow: 0 0 8px ${color};
        "></div>
        ${label ? `<span style="
          position: absolute;
          top: -18px;
          white-space: nowrap;
          font-size: 10px;
          font-weight: 700;
          color: #fff;
          background: rgba(15, 23, 42, 0.85);
          padding: 1px 4px;
          border-radius: 3px;
          border: 1px solid rgba(255,255,255,0.2);
        ">${label}</span>` : ''}
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });
};

const MAP_PROVIDERS = {
  esriDark: {
    name: 'Dark Canvas (Free)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; <a href="https://www.esri.com/">Esri</a> &mdash; Esri, DeLorme, NAVTEQ, HERE'
  },
  esriSatellite: {
    name: 'Satellite Ocean (Free)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; <a href="https://www.esri.com/">Esri</a> &mdash; Earthstar Geographics'
  },
  osm: {
    name: 'OpenStreetMap (Free)',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }
};

export default function MapTrackViewer({ historicalPoints = [], forecastPoints = [], cycloneName = "Storm" }) {
  const [selectedProvider, setSelectedProvider] = React.useState('esriDark');
  const [mapboxToken, setMapboxToken] = React.useState(import.meta.env.VITE_MAPBOX_TOKEN || '');
  const [showTokenInput, setShowTokenInput] = React.useState(false);

  const defaultCenter = historicalPoints.length > 0 
    ? [historicalPoints[historicalPoints.length - 1].lat, historicalPoints[historicalPoints.length - 1].lon]
    : [15.0, 85.0];

  // Coordinates for polylines
  const pastCoords = historicalPoints.map(p => [p.lat, p.lon]);
  
  // Forecast polyline starts from last historical point
  const lastPast = pastCoords.length > 0 ? pastCoords[pastCoords.length - 1] : null;
  const futureCoords = lastPast 
    ? [lastPast, ...forecastPoints.map(p => [p.lat, p.lon])]
    : forecastPoints.map(p => [p.lat, p.lon]);

  const currentTileUrl = mapboxToken && selectedProvider === 'mapbox'
    ? `https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/256/{z}/{x}/{y}@2x?access_token=${mapboxToken}`
    : MAP_PROVIDERS[selectedProvider]?.url || MAP_PROVIDERS.esriDark.url;

  const currentAttribution = mapboxToken && selectedProvider === 'mapbox'
    ? '&copy; <a href="https://www.mapbox.com/">Mapbox</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    : MAP_PROVIDERS[selectedProvider]?.attribution || MAP_PROVIDERS.esriDark.attribution;

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Track Forecast & 90% Confidence Cones</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Historical path (solid cyan) vs 24h AI-predicted track (dashed orange)
          </p>
        </div>
        
        {/* Layer & Mapbox Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
          <button
            onClick={() => setSelectedProvider('esriDark')}
            className={`btn-select-storm ${selectedProvider === 'esriDark' ? 'active' : ''}`}
            style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
          >
            Dark Canvas
          </button>
          <button
            onClick={() => setSelectedProvider('esriSatellite')}
            className={`btn-select-storm ${selectedProvider === 'esriSatellite' ? 'active' : ''}`}
            style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
          >
            Satellite
          </button>
          <button
            onClick={() => setSelectedProvider('osm')}
            className={`btn-select-storm ${selectedProvider === 'osm' ? 'active' : ''}`}
            style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
          >
            OSM
          </button>
          <button
            onClick={() => {
              if (!mapboxToken) {
                setShowTokenInput(!showTokenInput);
              } else {
                setSelectedProvider('mapbox');
              }
            }}
            className={`btn-select-storm ${selectedProvider === 'mapbox' ? 'active' : ''}`}
            style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
          >
            {mapboxToken ? 'Mapbox Dark' : '+ Mapbox Key'}
          </button>
        </div>
      </div>

      {/* Mapbox Token Input Bar if clicked */}
      {showTokenInput && (
        <div style={{
          background: 'rgba(56, 189, 248, 0.08)',
          border: '1px solid var(--border-highlight)',
          padding: '0.5rem 0.75rem',
          borderRadius: '6px',
          marginBottom: '0.75rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <input
            type="text"
            placeholder="Paste your Mapbox access token (pk.eyJ...)"
            value={mapboxToken}
            onChange={(e) => {
              setMapboxToken(e.target.value);
              if (e.target.value) setSelectedProvider('mapbox');
            }}
            style={{
              flex: 1,
              background: 'var(--bg-input)',
              border: '1px solid var(--border-color)',
              color: '#fff',
              padding: '0.3rem 0.5rem',
              borderRadius: '4px',
              fontSize: '0.78rem'
            }}
          />
          <button
            onClick={() => setShowTokenInput(false)}
            className="btn-primary"
            style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}
          >
            Apply
          </button>
        </div>
      )}

      <div className="map-wrapper">
        <MapContainer
          center={defaultCenter}
          zoom={5}
          scrollWheelZoom={true}
          style={{ height: '100%', width: '100%' }}
        >
          <ChangeView center={defaultCenter} zoom={5} />
          
          {/* Active Map Tile Layer */}
          <TileLayer
            key={selectedProvider + mapboxToken}
            attribution={currentAttribution}
            url={currentTileUrl}
          />

          {/* Historical Path Polyline */}
          {pastCoords.length > 1 && (
            <Polyline
              positions={pastCoords}
              pathOptions={{ color: '#38bdf8', weight: 3, opacity: 0.85 }}
            />
          )}

          {/* Forecast Path Polyline (Dashed) */}
          {futureCoords.length > 1 && (
            <Polyline
              positions={futureCoords}
              pathOptions={{ color: '#f59e0b', weight: 3, dashArray: '6, 8', opacity: 0.9 }}
            />
          )}

          {/* Uncertainty Circles (Confidence Cones) around Forecast Horizons */}
          {forecastPoints.map((pt, idx) => (
            <Circle
              key={`cone-${idx}`}
              center={[pt.lat, pt.lon]}
              radius={pt.uncertainty_radius_km * 1000} // convert km to meters
              pathOptions={{
                color: idx === 2 ? '#ef4444' : '#f59e0b',
                fillColor: idx === 2 ? '#ef4444' : '#f59e0b',
                fillOpacity: 0.12,
                weight: 1.5,
                dashArray: '4, 4'
              }}
            />
          ))}

          {/* Past Observation Markers */}
          {historicalPoints.map((pt, idx) => {
            const isLatest = idx === historicalPoints.length - 1;
            return (
              <Marker
                key={`hist-${idx}`}
                position={[pt.lat, pt.lon]}
                icon={createDotIcon(isLatest ? '#38bdf8' : '#64748b', isLatest ? 'CURRENT' : '', isLatest)}
              >
                <Popup>
                  <div style={{ color: '#0f172a', fontSize: '12px' }}>
                    <strong>{cycloneName}</strong><br />
                    Time: {pt.timestamp || `Step -${historicalPoints.length - 1 - idx}`}<br />
                    Lat: {pt.lat}°, Lon: {pt.lon}°<br />
                    Vmax: {pt.vmax} kt | MSLP: {pt.pres || 1000} hPa
                  </div>
                </Popup>
              </Marker>
            );
          })}

          {/* Forecast Horizon Markers */}
          {forecastPoints.map((pt, idx) => (
            <Marker
              key={`fc-${idx}`}
              position={[pt.lat, pt.lon]}
              icon={createDotIcon('#f59e0b', `+${pt.horizon}`)}
            >
              <Popup>
                <div style={{ color: '#0f172a', fontSize: '12px' }}>
                  <strong>Horizon: +{pt.horizon}</strong><br />
                  Coord: {pt.lat}°N, {pt.lon}°E<br />
                  Est. Vmax: {pt.predicted_vmax_kt} kt<br />
                  90% Uncertainty Radius: ±{pt.uncertainty_radius_km} km
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
        <span>Coordinates: WGS84 Datum</span>
        <span>Lead Horizons: 6h / 12h / 24h Synoptic Windows</span>
      </div>
    </div>
  );
}
