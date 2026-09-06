import React, { useState } from 'react';
import { ShieldAlert, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Zap } from 'lucide-react';

export default function RiskAuditPanel({ riskData }) {
  const [showFullAudit, setShowFullAudit] = useState(true);

  if (!riskData) return null;

  const {
    overall_risk_level = "Moderate",
    total_risk_score = 45,
    intensity_threat = "Severe",
    proximity_threat = "Coastal Corridor",
    rapid_intensification_risk = false,
    rule_audit_trace = []
  } = riskData;

  const getRiskBadgeClass = (level) => {
    switch (level.toLowerCase()) {
      case 'very high': return 'badge-very-strong';
      case 'high': return 'badge-strong';
      case 'moderate': return 'badge-moderate';
      default: return 'badge-weak';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'var(--accent-rose)';
    if (score >= 55) return 'var(--accent-amber)';
    if (score >= 30) return '#eab308';
    return 'var(--accent-cyan)';
  };

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldAlert size={20} color={getScoreColor(total_risk_score)} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Decision-Support Risk Index</h3>
        </div>
        <span className={`badge ${getRiskBadgeClass(overall_risk_level)}`} style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}>
          {overall_risk_level} Risk
        </span>
      </div>

      {/* Composite Score Bar */}
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Multi-Factor Threat Index</span>
          <span style={{ fontWeight: 700, color: getScoreColor(total_risk_score) }}>{total_risk_score} / 100</span>
        </div>
        <div style={{ height: 8, width: '100%', background: 'rgba(255,255,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${total_risk_score}%`,
            background: `linear-gradient(90deg, var(--accent-cyan), ${getScoreColor(total_risk_score)})`,
            transition: 'width 0.4s ease'
          }} />
        </div>
      </div>

      {/* Threat Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <div className="stat-box" style={{ padding: '0.75rem' }}>
          <div className="stat-label">Wind Field Hazard</div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
            {intensity_threat}
          </div>
        </div>

        <div className="stat-box" style={{ padding: '0.75rem' }}>
          <div className="stat-label">Coastal Impact Perimeter</div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
            {proximity_threat}
          </div>
        </div>
      </div>

      {/* Rapid Intensification Alert Banner if triggered */}
      {rapid_intensification_risk && (
        <div style={{
          background: 'rgba(244, 63, 94, 0.12)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          padding: '0.65rem 0.85rem',
          borderRadius: '6px',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          color: '#fb7185',
          fontSize: '0.82rem',
          fontWeight: 600,
          marginBottom: '1rem'
        }}>
          <Zap size={16} />
          Rapid Intensification (RI) Warning: Expected +20kt spike in 24h lead window
        </div>
      )}

      {/* Transparent Rule Audit Log */}
      <div>
        <button
          onClick={() => setShowFullAudit(!showFullAudit)}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            fontSize: '0.82rem',
            fontWeight: 600,
            cursor: 'pointer',
            padding: '0.35rem 0'
          }}
        >
          <span>Transparent Rule Audit Trail ({rule_audit_trace.length} criteria)</span>
          {showFullAudit ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showFullAudit && (
          <div className="rule-trace-list">
            {rule_audit_trace.map((r, i) => (
              <div key={`rule-${i}`} className={`rule-trace-item ${r.triggered ? 'triggered' : ''}`}>
                <div>
                  <div style={{ fontWeight: 600, color: r.triggered ? '#fb7185' : 'var(--text-secondary)' }}>
                    {r.rule_id}: {r.condition}
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '2px' }}>
                    {r.rationale}
                  </div>
                </div>
                <div style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                  <span style={{
                    fontWeight: 700,
                    color: r.triggered ? 'var(--accent-rose)' : 'var(--text-muted)',
                    fontSize: '0.85rem'
                  }}>
                    {r.triggered ? `+${r.risk_score_impact} pts` : '0 pts'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
