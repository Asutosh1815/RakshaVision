# 🛡️ RakshaVision - Industrial AI Safety & Hazard Monitoring System

**RakshaVision** is an AI-powered, real-time industrial safety monitoring platform engineered for manufacturing plants, construction sites, and chemical facilities. It automates compliance verification for Personal Protective Equipment (PPE) and provides sub-second optical detection of early fire and smoke hazards from CCTV and recorded video streams.

---

## 🌟 Key Capabilities

1. **Multi-Item PPE Compliance Detection**:
   - **Hard Hats & Safety Helmets**: Dual-ensemble YOLO detection with spatial head-zone binding.
   - **High-Visibility Safety Vests**: Fluorescent chromaticity analysis (Neon Lime-Yellow and Neon Orange) combined with retroreflective stripe recognition.
   - **Safety Footwear & Boots**: Feet-level PPE association.
   - **Safety Gloves**: Hand-region spatial PPE verification.
   - **Individual Worker Status**: Assigns persistent IDs, maps gear to worker anatomical zones, and classifies status (`COMPLIANT`, `VIOLATION`, `CAUTION`).

2. **Early Smoke & Fire Hazard Intervention**:
   - Sub-second detection of active flames and expanding smoke plumes.
   - Multi-spectral chromatic rules ($RGB + HSV + YCrCb$) with Non-Maximum Suppression (NMS) to eliminate false alarms from industrial yellow safety markings.
   - Provides up to **4.5 minutes faster response** compared to traditional heat/smoke aspirating ceiling sensors.

3. **Multi-Camera & Location Context**:
   - Pre-configured factory zones (*Bay 1: Machining*, *Bay 2: Welding & Chemical*, *Bay 3: Logistics Dock*, *Control Room*).
   - Configurable safety policies per zone (e.g. mandatory gloves in hot-work areas).

4. **Rapid Alert Dispatch & Incident Audit Trail**:
   - High-visibility Heads-Up Display (HUD) overlays with live compliance scoring.
   - Timestamped incident logging with cropped incident snapshot evidence.
   - One-click CSV audit log export.

5. **Flexible Input Modes**:
   - Continuous CCTV simulation feed.
   - Video file upload (`MP4`, `AVI`, `MOV`).
   - Deep single-image inspection with individual worker zoom.
   - Live connected webcam.

---

## 🏗️ System Architecture

```
RakshaVision/
├── app.py                     # Streamlit Command Center UI
├── core/
│   ├── types.py               # Data models & Enums (BoundingBox, WorkerCompliance, etc.)
│   ├── detector.py            # YOLO PPE & person detector, high-vis vest analyzer
│   ├── hazard_detector.py     # Real-time Fire & Smoke detection engine
│   ├── compliance_engine.py   # Spatial anatomical mapping & zone policy logic
│   ├── zone_manager.py        # Location profiles & camera configuration
│   ├── visualizer.py          # Real-time CCTV HUD overlay renderer
│   └── logger.py              # Incident dispatch & CSV audit logger
├── demo_assets/               # Pre-bundled industrial demo videos and images
├── weights/                   # YOLOv8 weights (base, PPE, hardhat)
├── snapshots/                 # Incident snapshot thumbnail evidence
├── tests/
│   └── test_pipeline.py       # End-to-end automated pipeline tests
└── requirements.txt           # Project dependencies
```

---

## 🚀 Getting Started

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python tests/test_pipeline.py
```

### 3. Launch the Command Center
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the RakshaVision Command Center.

---

## 🧪 Demonstration Scenarios Included

RakshaVision comes pre-packaged with sample industrial scenarios ready for instant testing:
- **Scenario 1 - Compliant Workers**: Workers wearing hard hats and high-vis vests.
- **Scenario 2 - Safety Violations**: Workers entering dangerous zones without required gear.
- **Scenario 3 - Fire & Smoke Incident**: Early flame and smoke outbreak trigger.
- **CCTV Video Simulation**: Dynamic multi-worker feed demonstrating real-time tracking, live compliance rate calculations, and emergency alert banners.
