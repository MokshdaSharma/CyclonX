import React, { useState, useEffect, useRef } from 'react';
import { Layers, Eye, Sliders, Info } from 'lucide-react';

export default function SatelliteViewer({
  gradcamBase64 = null,
  cycloneId = "STORM",
  predictedVmax = 75,
  intensityCategory = "Strong"
}) {
  const [selectedChannel, setSelectedChannel] = useState('IR1');
  const [showGradcam, setShowGradcam] = useState(true);
  const [alpha, setAlpha] = useState(0.45);
  const canvasRef = useRef(null);

  // Generate synthetic satellite channel visualization if raw image stream is simulating
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Base background
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H / 2;

    // Draw realistic vortex cloud spiral bands
    for (let r = 10; r < 90; r += 2) {
      const angleOffset = r * 0.15;
      const alphaVal = Math.max(0.1, 1.0 - (r / 95));

      if (selectedChannel === 'IR1') {
        // Infrared: Cold cloud tops (white/cyan) vs warm ocean (dark navy)
        ctx.strokeStyle = `rgba(210, 235, 255, ${alphaVal * 0.65})`;
      } else if (selectedChannel === 'WV') {
        // Water Vapor: Mid-troposphere moisture (amber/cyan)
        ctx.strokeStyle = `rgba(56, 189, 248, ${alphaVal * 0.75})`;
      } else if (selectedChannel === 'PMW') {
        // Microwave: Deep convective rainbands (vibrant green/orange)
        ctx.strokeStyle = `rgba(52, 211, 153, ${alphaVal * 0.8})`;
      } else {
        // False color composite
        ctx.strokeStyle = `rgba(147, 197, 253, ${alphaVal * 0.7})`;
      }

      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r, angleOffset, angleOffset + Math.PI * 1.4);
      ctx.stroke();
    }

    // Draw eye region
    ctx.fillStyle = '#030712';
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();

  }, [selectedChannel]);

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Multi-Source Satellite & Explainability</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            TCIR 201x201 Channels (IR1, WV, PMW) + Grad-CAM Feature Attribution
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.35rem' }}>
          {['IR1', 'WV', 'PMW', 'Composite'].map((ch) => (
            <button
              key={ch}
              onClick={() => setSelectedChannel(ch)}
              className={`btn-select-storm ${selectedChannel === ch ? 'active' : ''}`}
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}
            >
              {ch}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
        {/* Image Canvas with Overlay */}
        <div className="image-canvas-wrapper" style={{ position: 'relative', width: 260, height: 260 }}>
          {/* Base Channel Canvas */}
          <canvas
            ref={canvasRef}
            width={201}
            height={201}
            style={{ width: '100%', height: '100%', display: 'block', borderRadius: '8px' }}
          />

          {/* Grad-CAM Heatmap Image Overlay */}
          {showGradcam && gradcamBase64 && (
            <img
              src={gradcamBase64}
              alt="Grad-CAM Overlay"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                opacity: alpha,
                pointerEvents: 'none',
                mixBlendMode: 'screen',
                transition: 'opacity 0.15s ease'
              }}
            />
          )}

          <div style={{
            position: 'absolute',
            bottom: 8,
            left: 8,
            fontSize: '11px',
            background: 'rgba(0,0,0,0.7)',
            padding: '2px 6px',
            borderRadius: '4px',
            color: '#94a3b8'
          }}>
            Channel: {selectedChannel} (201x201)
          </div>
        </div>

        {/* Grad-CAM Controls */}
        <div style={{ width: '100%', maxWidth: '340px', background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.82rem', fontWeight: 600, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showGradcam}
                onChange={(e) => setShowGradcam(e.target.checked)}
                style={{ accentColor: 'var(--accent-cyan)' }}
              />
              Grad-CAM Visual Attention
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Opacity: {Math.round(alpha * 100)}%
            </span>
          </div>

          <div className="slider-container">
            <Sliders size={14} color="var(--text-muted)" />
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={alpha}
              disabled={!showGradcam}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="custom-range"
            />
          </div>

          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            Attribution highlights convection gradients around eyewall driving the {predictedVmax} kt estimation.
          </p>
        </div>
      </div>
    </div>
  );
}
