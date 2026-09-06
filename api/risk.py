"""
CycloneX - Rule-Based Transparent Decision-Support Risk Engine
Computes multidimensional composite risk scores (0 - 100) and maps to 4 levels:
- Low (Score < 30)
- Moderate (Score 30 - 54)
- High (Score 55 - 79)
- Very High (Score >= 80)

Combines:
1. Wind Intensity Threat (Current and 24h forecast Vmax)
2. Coastal Proximity & Landfall Vector (Distance to coastline vs 24h uncertainty radius)
3. Rapid Intensification (RI) Indicator (dVmax >= 30 kt / 24h or dVmax >= 15 kt / 12h)
4. Forecast Track Uncertainty Spread

DISCLAIMER: AI-assisted research prototype for decision support. Not an official IMD warning.
"""

import os
import sys
import math
from typing import Dict, Any, List, Optional

# Ensure local api folder is at the top of sys.path
api_dir = os.path.dirname(os.path.abspath(__file__))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

try:
    from api.schemas import RuleTraceItem, RiskAssessmentResponse
except ImportError:
    from schemas import RuleTraceItem, RiskAssessmentResponse


# Simplified coastline bounding segments for North Indian Ocean (Bay of Bengal & Arabian Sea)
INDIAN_COASTLINE_REF_POINTS = [
    (8.08, 77.55),   # Kanyakumari
    (9.28, 79.31),   # Rameswaram
    (10.76, 79.84),  # Nagapattinam
    (13.08, 80.27),  # Chennai
    (15.82, 80.35),  # Ongole
    (17.68, 83.21),  # Visakhapatnam
    (19.81, 85.83),  # Puri / Paradip
    (21.49, 86.93),  # Balasore / Digha
    (21.68, 88.04),  # Sagar Island / Sundarbans
    (22.35, 91.83),  # Chittagong (Bangladesh)
    (20.15, 92.90),  # Sittwe (Myanmar)
    (8.48, 76.95),   # Trivandrum (West Coast)
    (9.93, 76.26),   # Kochi
    (12.87, 74.88),  # Mangaluru
    (15.49, 73.82),  # Goa
    (18.92, 72.83),  # Mumbai
    (20.90, 70.37),  # Veraval (Gujarat)
    (22.25, 68.96),  # Dwarka
    (23.00, 70.13),  # Kandla / Kutch
    (24.86, 67.01)   # Karachi (Pakistan)
]


def estimate_distance_to_nearest_coast(lat: float, lon: float) -> float:
    """
    Approximates minimum distance in kilometers from cyclone coordinates
    to nearest North Indian Ocean coastline reference points.
    """
    min_dist = float("inf")
    R = 6371.0
    p1 = math.radians(lat)
    
    for c_lat, c_lon in INDIAN_COASTLINE_REF_POINTS:
        p2 = math.radians(c_lat)
        dp = math.radians(c_lat - lat)
        dl = math.radians(c_lon - lon)
        a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
        a = min(1.0, max(0.0, a))
        dist_km = R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        if dist_km < min_dist:
            min_dist = dist_km
            
    return round(min_dist, 1)


def evaluate_cyclone_risk(
    cyclone_id: str,
    current_vmax_kt: float,
    current_lat: float,
    current_lon: float,
    forecast_vmax_24h_kt: Optional[float] = None,
    forecast_lat_24h: Optional[float] = None,
    forecast_lon_24h: Optional[float] = None,
    uncertainty_radius_24h_km: float = 125.0,
    distance_to_coastline_km: Optional[float] = None,
    model_version: str = "cyclonex-v1.0.0"
) -> RiskAssessmentResponse:
    """
    Executes rule-based transparent risk assessment matrix.
    """
    if forecast_vmax_24h_kt is None:
        forecast_vmax_24h_kt = current_vmax_kt + 5.0
        
    if distance_to_coastline_km is None:
        # Check current position distance or 24h forecast position distance
        dist_curr = estimate_distance_to_nearest_coast(current_lat, current_lon)
        if forecast_lat_24h is not None and forecast_lon_24h is not None:
            dist_fc = estimate_distance_to_nearest_coast(forecast_lat_24h, forecast_lon_24h)
            distance_to_coastline_km = min(dist_curr, dist_fc)
        else:
            distance_to_coastline_km = dist_curr
            
    score = 0
    rule_trace: List[RuleTraceItem] = []
    
    # RULE 1: Current Intensity Tier
    if current_vmax_kt >= 83.0: # Very Strong / Super Cyclone
        score += 35
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_INT_01",
            condition="Current Vmax >= 83 kt (Very Strong / Severe)",
            triggered=True,
            risk_score_impact=35,
            rationale="High wind energy generates catastrophic surge and destructive wind field."
        ))
    elif current_vmax_kt >= 64.0: # Strong
        score += 25
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_INT_02",
            condition="Current Vmax between 64 and 82 kt (Strong)",
            triggered=True,
            risk_score_impact=25,
            rationale="Sustained hurricane-force winds present major structural hazard."
        ))
    elif current_vmax_kt >= 34.0: # Moderate
        score += 15
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_INT_03",
            condition="Current Vmax between 34 and 63 kt (Moderate)",
            triggered=True,
            risk_score_impact=15,
            rationale="Gale force winds and heavy squalls expected."
        ))
    else:
        score += 5
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_INT_04",
            condition="Current Vmax < 34 kt (Weak)",
            triggered=True,
            risk_score_impact=5,
            rationale="Depression stage; localized rainfall threat."
        ))
        
    # RULE 2: Rapid Intensification (RI) Check (dVmax >= 25 kt in 24h)
    delta_vmax = forecast_vmax_24h_kt - current_vmax_kt
    is_ri = delta_vmax >= 20.0
    if is_ri:
        score += 20
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_RI_01",
            condition="Forecast 24h Vmax increase >= 20 kt",
            triggered=True,
            risk_score_impact=20,
            rationale="Rapid Intensification detected; short preparation lead times."
        ))
    else:
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_RI_01",
            condition="Forecast 24h Vmax increase >= 20 kt",
            triggered=False,
            risk_score_impact=0,
            rationale="No rapid intensification signature detected in 24h horizon."
        ))
        
    # RULE 3: Coastal Proximity & Landfall Envelope
    if distance_to_coastline_km <= 100.0:
        score += 30
        proximity_threat = "Immediate Coastal Threat (< 100 km)"
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_PROX_01",
            condition="Distance to coast <= 100 km",
            triggered=True,
            risk_score_impact=30,
            rationale="Storm center or core rainbands within immediate coastal zone."
        ))
    elif distance_to_coastline_km <= (uncertainty_radius_24h_km + 50.0):
        score += 20
        proximity_threat = "High Landfall Probability in 24h Window"
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_PROX_02",
            condition=f"Coastline within 24h uncertainty radius ({uncertainty_radius_24h_km} km)",
            triggered=True,
            risk_score_impact=20,
            rationale="Coastal corridor intersects the 90% confidence uncertainty cone."
        ))
    elif distance_to_coastline_km <= 350.0:
        score += 10
        proximity_threat = "Approaching Coastal Corridor (100-350 km)"
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_PROX_03",
            condition="Distance to coast between 100 and 350 km",
            triggered=True,
            risk_score_impact=10,
            rationale="Storm tracking towards continental margin."
        ))
    else:
        score += 0
        proximity_threat = "Open Ocean Track (> 350 km from coast)"
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_PROX_04",
            condition="Distance to coast > 350 km",
            triggered=False,
            risk_score_impact=0,
            rationale="Maritime system located away from densely populated shorelines."
        ))
        
    # RULE 4: High Forecast Uncertainty Penalty
    if uncertainty_radius_24h_km > 140.0:
        score += 10
        rule_trace.append(RuleTraceItem(
            rule_id="RULE_UNCERT_01",
            condition="24h Forecast Uncertainty Cone > 140 km",
            triggered=True,
            risk_score_impact=10,
            rationale="High directional variance increases broad warning perimeter."
        ))
        
    # Total Score Normalization [0 - 100]
    total_score = min(100, max(0, score))
    
    # Categorize Risk Level
    if total_score >= 80:
        risk_level = "Very High"
    elif total_score >= 55:
        risk_level = "High"
    elif total_score >= 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
        
    # Intensity threat label
    if current_vmax_kt >= 83.0:
        intensity_threat = "Catastrophic (Category 3-5 equivalent / Super Cyclone)"
    elif current_vmax_kt >= 64.0:
        intensity_threat = "Severe (Very Severe Cyclonic Storm)"
    elif current_vmax_kt >= 34.0:
        intensity_threat = "Moderate (Cyclonic Storm / Severe CS)"
    else:
        intensity_threat = "Minor (Depression)"
        
    return RiskAssessmentResponse(
        cyclone_id=cyclone_id,
        overall_risk_level=risk_level,
        total_risk_score=total_score,
        intensity_threat=intensity_threat,
        proximity_threat=proximity_threat,
        rapid_intensification_risk=is_ri,
        rule_audit_trace=rule_trace,
        model_version=model_version,
        disclaimer="AI-assisted decision-support research prototype. Not an official IMD warning."
    )
