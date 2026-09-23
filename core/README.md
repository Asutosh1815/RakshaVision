# 🧠 Core AI & Computer Vision Engine

The `core/` package forms the architectural backbone of **RakshaVision AI**, containing all deep learning models, optical heuristics, compliance arbitration, and hazard detection logic.

## Modules

| File | Responsibilities |
| :--- | :--- |
| `types.py` | Core data structures including `BoundingBox`, `Detection`, `WorkerCompliance`, `ComplianceStatus`, `ZoneConfig`, and `ActionableAlert`. |
| `detector.py` | Multi-model YOLOv8 inference wrapper (`SafetyGearDetector`), per-worker padded crop inference (`detect_ppe_for_person`), retroreflective silver-tape Sobel analyzer, and optical footwear/glove fallbacks. |
| `compliance_engine.py` | Spatial IoU arbitration (`ComplianceEngine`) evaluating 4-point PPE (Helmets, Safety Vests, Boots, Gloves) against active zone policies. |
| `hazard_detector.py` | Zero-latency combustion detector with multi-spectral steam ($V>238, S<14$) and airborne dust ($\sigma>47.0$) rejection, plus worker-body decoupling. |
| `alert_router.py` | Role-based Standard Operating Procedure (SOP) dispatch engine routing actionable alerts to Fire Marshals, Safety Supervisors, EHS Directors, and Zone Officers. |
| `zone_manager.py` | Facility zone configuration manager mapping camera coordinates to customized PPE compliance rules. |
| `visualizer.py` | Real-time video overlay HUD renderer displaying clean status badges (`[+] Helmet`, `[+] Vest`, `[+] Boots`, `[+] Gloves`) and latency telemetry. |
| `logger.py` | Thread-safe compliance audit and incident evidence logging to timestamped CSV files. |
