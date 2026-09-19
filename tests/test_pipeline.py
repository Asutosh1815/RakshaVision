"""
Automated Pipeline Verification Test for RakshaVision.
Tests detector, compliance engine, hazard detection, visualizer HUD, and incident logger.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np

from core.detector import SafetyGearDetector
from core.hazard_detector import HazardDetector
from core.compliance_engine import ComplianceEngine
from core.visualizer import SafetyVisualizer
from core.logger import IncidentLogger
from core.zone_manager import ZoneManager
from core.types import ComplianceStatus, HazardType


def test_full_pipeline():
    print("=== [TEST 1] Initializing Modules ===")
    detector = SafetyGearDetector()
    hazard_det = HazardDetector()
    compliance_engine = ComplianceEngine(detector)
    visualizer = SafetyVisualizer()
    logger = IncidentLogger(snapshot_dir="snapshots_test")
    zone_mgr = ZoneManager()
    zone = zone_mgr.get_zone("ZONE_FAB_BAY")
    assert zone is not None, "Failed to fetch zone configuration"
    print("✓ All modules successfully initialized.")

    print("\n=== [TEST 2] Testing Worker & PPE Detection on Scenario Violation ===")
    img_violation = cv2.imread("demo_assets/scenario_violation.jpg")
    assert img_violation is not None, "Failed to load scenario_violation.jpg"

    # Even on synthetic workers or real workers, run detector
    persons = detector.detect_persons(img_violation, conf_threshold=0.10)
    raw_ppe = detector.detect_ppe_items(img_violation, conf_threshold=0.10)
    print(f"Persons detected: {len(persons)}, Raw PPE items detected: {len(raw_ppe)}")

    workers = compliance_engine.evaluate_frame(img_violation, persons, raw_ppe, zone)
    metrics = compliance_engine.calculate_site_metrics(workers)
    print(f"Workers evaluated: {len(workers)}, Site metrics: {metrics}")
    print("✓ Worker & compliance evaluation passed.")

    print("\n=== [TEST 3] Testing Fire & Smoke Hazard Detection ===")
    img_fire = cv2.imread("demo_assets/scenario_fire_hazard.jpg")
    assert img_fire is not None, "Failed to load scenario_fire_hazard.jpg"

    hazards = hazard_det.detect_hazards(img_fire)
    print(f"Hazards detected in fire scene: {len(hazards)}")
    for h in hazards:
        print(f"  - Hazard: {h.hazard_type.value}, Conf: {h.confidence}, Box: {h.box.to_int_tuple()}")
    
    # Verify fire or smoke detected
    has_fire_or_smoke = any(h.hazard_type in (HazardType.FIRE, HazardType.SMOKE) for h in hazards)
    assert has_fire_or_smoke, "Hazard detector failed to flag flame/smoke in fire scene!"
    print("✓ Hazard detection passed.")

    print("\n=== [TEST 4] Testing Incident Logger & CSV Export ===")
    logged_hazards = logger.log_hazard_incidents(img_fire, hazards, zone)
    print(f"Logged hazard incidents: {len(logged_hazards)}")
    assert len(logged_hazards) > 0, "Logger failed to record hazard incident"

    csv_path = logger.export_csv("snapshots_test/test_audit_log.csv")
    assert os.path.exists(csv_path), "CSV export failed"
    print(f"✓ Incident logger & CSV export passed ({os.path.getsize(csv_path)} bytes).")

    print("\n=== [TEST 5] Testing HUD Visualizer ===")
    hud_frame = visualizer.draw_hud(img_fire, workers, hazards, zone, fps=28.5)
    assert hud_frame.shape == img_fire.shape, "Visualizer changed frame dimensions"
    cv2.imwrite("snapshots_test/test_rendered_hud.jpg", hud_frame)
    print("✓ HUD Visualizer passed, output saved to snapshots_test/test_rendered_hud.jpg.")

    print("\n=== ALL PIPELINE TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    test_full_pipeline()
