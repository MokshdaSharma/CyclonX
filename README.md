# CycloneX — AI/ML Tropical Cyclone Identification, Classification & Track Prediction

## 🌪️ System Overview

CycloneX provides an end-to-end multimodal pipeline for tropical cyclone analysis in the North Indian Ocean (Bay of Bengal & Arabian Sea) and global basins:
1. **Multi-Source Satellite Processing**: Ingests Infrared (IR1), Water Vapor (WV), and Passive Microwave (PMW) channels (201×201 resolution) from the TCIR dataset, applying training-split robust normalization.
2. **Deep Intensity Estimation**: Transfer-learning with EfficientNetB0 (Huber loss, learning rate scheduling) estimating maximum sustained surface wind ($V_{max}$) mapped to 4 IMD-aligned intensity classes: Weak (<34 kt), Moderate (34–63 kt), Strong (64–82 kt), and Very Strong (≥83 kt).
3. **Visual Explainability (Grad-CAM)**: Generates feature activation heatmaps over the IR1 brightness temperature channel to highlight convective eyewall gradients driving intensity predictions.
4. **Temporal Track & Intensity Forecasting**: Stacked GRU and LSTM networks predicting 6h, 12h, and 24h coordinates ($\Delta\text{lat}, \Delta\text{lon}$) and intensity ($V_{max}, \text{MSLP}$), benchmarked against a persistence baseline.
5. **Multimodal Fusion & Uncertainty Estimation**: Merges visual embeddings and temporal recurrent hidden states with Monte Carlo (MC) Dropout to construct calibrated 90% confidence uncertainty cones.
6. **Transparent Rule-Based Risk Engine**: Evaluates a 0–100 composite risk index with an auditable rule trace factoring in wind field hazard, coastal proximity, and rapid intensification (RI) signatures.
7. **Interactive Dashboard**: Modern dark-theme glassmorphism interface with OpenStreetMap/Leaflet track rendering, satellite composite inspection, intensity trend charts, and real-time risk diagnostics.

---

## 📁 Repository Structure

```
cyclonex/
├── configs/
│   ├── preprocessing_config.json    # Channel stats (median, IQR) & normalization bounds
│   └── intensity_categories.json    # 4 IMD/Saffir-Simpson intensity thresholds
├── data/
│   ├── download_tcir.py             # TCIR-2017 HDF5 loader & synthetic dataset generator
│   ├── preprocess.py                # Robust scaling & leak-free cyclone-level splitting
│   └── track_dataset.py             # Best-track CSV ingestion, sliding windows & Haversine math
├── models/
│   ├── train_intensity_model.py     # EfficientNetB0 regression training pipeline
│   ├── train_track_model.py         # LSTM vs GRU vs Persistence baseline comparison
│   ├── fusion_model.py              # Visual-temporal multimodal fusion & MC-dropout
│   ├── gradcam.py                   # Grad-CAM heatmap generator & IR1 overlay
│   └── evaluate.py                  # MAE, RMSE, R2, F1, and Haversine distance metrics
├── api/
│   ├── app.py                       # FastAPI REST backend with CORS & demo presets
│   ├── schemas.py                   # Pydantic V2 request & response models
│   ├── inference.py                 # Multi-model inference runner & version tagging
│   ├── risk.py                      # Rule-based decision-support risk scoring engine
│   ├── Dockerfile                   # Production container for Hugging Face Spaces
│   └── requirements.txt             # API-specific dependencies
├── dashboard/                       # React 18 + Vite SPA with Leaflet & SVG charts
│   ├── src/
│   │   ├── components/              # MapTrackViewer, SatelliteViewer, IntensityChart, RiskAuditPanel
│   │   ├── App.jsx                  # Main dashboard layout
│   │   └── index.css                # Glassmorphic dark styling system
│   ├── package.json
│   └── vite.config.js
├── notebooks/
│   └── demo_dashboard.ipynb         # Interactive Google Colab / Kaggle demonstration
├── tests/
│   ├── test_preprocessing.py        # Robust scaling & zero-leakage split tests
│   ├── test_api.py                  # FastAPI endpoint integration tests
│   └── test_models.py               # Grad-CAM, Haversine, and risk rule tests
├── .github/workflows/
│   └── ci.yml                       # Continuous Integration workflow
├── requirements.txt                 # Root Python dependencies
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Backend Setup & Local Server

```bash
# Clone repository
git clone https://github.com/cyclonex/cyclonex.git
cd cyclonex

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest -v tests/

# Launch FastAPI backend
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
API documentation is available at `http://localhost:8000/docs`.

### 2. Frontend Dashboard Setup

```bash
cd dashboard

# Install npm dependencies
npm install

# Start Vite dev server
npm run dev
```
Open `http://localhost:3000` in your browser.

---


---

## 🛡️ License

Released under the **MIT License**. Created for research and disaster risk reduction education.
