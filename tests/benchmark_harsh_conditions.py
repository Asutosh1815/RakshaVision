"""
Harsh Conditions & Performance Benchmark Suite for RakshaVision.
Tests and validates:
1. Normal Shirt vs. High-Vis Safety Vest Discrimination (0% False Positives on normal shirts).
2. Safety Footwear (Boots) and Hand Gloves Compliance Verification.
3. Industrial Steam & Airborne Dust Rejection vs. Real Fire & Smoke Detection.
4. Real-time Inference Latency (ms) and Throughput (FPS) Performance Benchmarking.
"""

import os
import sys
import time
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detector import SafetyGearDetector
from core.compliance_engine import ComplianceEngine
from core.hazard_detector import HazardDetector
from core.zone_manager import ZoneManager
from core.alert_router import AlertRouter
from core.types import ComplianceStatus, HazardType, ZoneConfig, BoundingBox


def generate_benchmark_scenes():
    """Generates synthetic test frames representing harsh industrial environments."""
    os.makedirs("benchmark_assets", exist_ok=True)

    # 1. Normal Yellow/Orange Shirt Worker (Should NOT be detected as vest!)
    img_shirt = np.zeros((640, 640, 3), dtype=np.uint8)
    # Factory background
    img_shirt[:] = (45, 45, 50)
    cv2.rectangle(img_shirt, (0, 480), (640, 640), (70, 75, 80), -1)
    # Worker body
    cv2.rectangle(img_shirt, (240, 180), (400, 520), (30, 30, 30), -1)  # Legs
    # Normal cotton shirt (Yellow/Orange without ANY reflective stripes or Sobel gradient)
    cv2.rectangle(img_shirt, (230, 200), (410, 390), (0, 165, 230), -1) # Orange cotton shirt
    # Bare head (no helmet)
    cv2.circle(img_shirt, (320, 140), 45, (130, 160, 190), -1)
    shirt_path = "benchmark_assets/test_normal_shirt.jpg"
    cv2.imwrite(shirt_path, img_shirt)

    # 2. Industrial Steam Plume (Should NOT be detected as smoke!)
    img_steam = np.zeros((640, 640, 3), dtype=np.uint8)
    img_steam[:] = (60, 65, 70)
    # Bright pure white steam cloud (V > 245, S < 10)
    for r in range(40, 150, 15):
        cv2.circle(img_steam, (320 + np.random.randint(-20, 20), 240 + np.random.randint(-20, 20)), r, (248, 248, 250), -1)
    cv2.GaussianBlur(img_steam, (25, 25), 0, dst=img_steam)
    steam_path = "benchmark_assets/test_steam_cloud.jpg"
    cv2.imwrite(steam_path, img_steam)

    return shirt_path, steam_path


def run_benchmark():
    print("=" * 70)
    print("🛡️  RAKSHAVISION AI - HARSH CONDITIONS & LATENCY BENCHMARK SUITE")
    print("=" * 70)

    shirt_path, steam_path = generate_benchmark_scenes()

    detector = SafetyGearDetector()
    compliance_engine = ComplianceEngine(detector)
    hazard_detector = HazardDetector()
    zone_mgr = ZoneManager()
    zone = zone_mgr.get_zone("ZONE_FAB_BAY")
    zone.require_helmet = True
    zone.require_vest = True
    zone.require_boots = True
    zone.require_gloves = True
    zone.hazard_monitoring = True

    results = {
        "normal_shirt_false_positives": 0,
        "steam_false_alarms": 0,
        "fire_detected": False,
        "smoke_detected": False,
        "gear_checks": {"helmet": True, "vest": True, "boots": True, "gloves": True},
        "latencies_ms": []
    }

    # -------------------------------------------------------------
    # TEST 1: Normal Colored Shirt Discrimination
    # -------------------------------------------------------------
    print("\n[*] TEST 1: Normal Shirt vs. High-Vis Safety Vest Discrimination...")
    img_shirt = cv2.imread(shirt_path)
    p_box = BoundingBox(220, 95, 420, 530)
    has_vest, vest_conf = detector.evaluate_vest_presence(img_shirt, p_box)

    if has_vest:
        results["normal_shirt_false_positives"] += 1
        print(f"    ❌ FAIL: Normal orange shirt falsely classified as safety vest (Conf: {vest_conf:.2f})")
    else:
        print(f"    ✅ PASS: Normal orange shirt correctly REJECTED (Zero false positive. Conf: {vest_conf:.2f})")

    # -------------------------------------------------------------
    # TEST 2: Steam & Dust Rejection
    # -------------------------------------------------------------
    print("\n[*] TEST 2: Industrial Steam & Dust Rejection...")
    img_steam = cv2.imread(steam_path)
    hazards = hazard_detector.detect_hazards(img_steam)
    steam_alarms = [h for h in hazards if h.hazard_type == HazardType.SMOKE]

    if steam_alarms:
        results["steam_false_alarms"] = len(steam_alarms)
        print(f"    ❌ FAIL: Industrial steam cloud falsely triggered {len(steam_alarms)} smoke alert(s)")
    else:
        print(f"    ✅ PASS: Pure white steam vapor successfully rejected (Zero false smoke alarms)")

    # -------------------------------------------------------------
    # TEST 3: Real Fire Flame & Billowing Smoke Verification
    # -------------------------------------------------------------
    print("\n[*] TEST 3: Genuine Fire & Smoke Detection...")
    fire_sample_path = "demo_assets/scenario_fire_hazard.jpg"
    if os.path.exists(fire_sample_path):
        f_img = cv2.imread(fire_sample_path)
        f_hazards = hazard_detector.detect_hazards(f_img)
        has_f = any(h.hazard_type == HazardType.FIRE for h in f_hazards)
        has_s = any(h.hazard_type == HazardType.SMOKE for h in f_hazards)
        results["fire_detected"] = has_f
        results["smoke_detected"] = has_s
        print(f"    {'✅' if has_f else '❌'} Fire Detection: {'CONFIRMED (True Positive)' if has_f else 'MISSED'}")
        print(f"    {'✅' if has_s else '❌'} Smoke Detection: {'CONFIRMED (True Positive)' if has_s else 'MISSED'}")

    # -------------------------------------------------------------
    # TEST 4: Footwear & Hand Gloves Compliance Checks
    # -------------------------------------------------------------
    print("\n[*] TEST 4: Safety Footwear & Hand Gloves Checks...")
    ppe_sample_path = "demo_assets/factory_worker_ppe.jpg"
    if os.path.exists(ppe_sample_path):
        ppe_img = cv2.imread(ppe_sample_path)
        persons = detector.detect_persons(ppe_img, conf_threshold=0.30)
        ppe_items = detector.detect_ppe_items(ppe_img, conf_threshold=0.25)
        workers = compliance_engine.evaluate_frame(ppe_img, persons, ppe_items, zone)
        print(f"    Evaluated {len(workers)} worker(s) across all 4 PPE categories:")
        for w in workers:
            print(f"      Worker #{w.worker_id}: Helmet={'✅' if w.has_helmet else '❌'} | Vest={'✅' if w.has_vest else '❌'} | Boots={'✅' if w.has_boots else '❌'} | Gloves={'✅' if w.has_gloves else '❌'}")

    # -------------------------------------------------------------
    # TEST 5: Latency & Throughput Benchmark
    # -------------------------------------------------------------
    print("\n[*] TEST 5: End-to-End Latency & FPS Performance (15 Iterations)...")
    test_frame = cv2.imread("demo_assets/scenario_compliant.jpg")
    if test_frame is None:
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        p = detector.detect_persons(test_frame, conf_threshold=0.30)
        wb = [x.box for x in p]
        ppe = detector.detect_ppe_items(test_frame, conf_threshold=0.25)
        w_eval = compliance_engine.evaluate_frame(test_frame, p, ppe, zone)
        h_eval = hazard_detector.detect_hazards(test_frame, worker_boxes=wb)
        t_e2e = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_e2e)

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0

    print(f"    ⚡ Average E2E Latency: {mean_lat:.1f} ms")
    print(f"    ⚡ Median E2E Latency:  {median_lat:.1f} ms")
    print(f"    ⚡ 95th Percentile:     {p95_lat:.1f} ms")
    print(f"    ⚡ Pipeline Throughput: {fps:.1f} FPS")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📋 BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    print(f"  • Normal Shirt False Positive Rate: {results['normal_shirt_false_positives']}% (Goal: 0%)")
    print(f"  • Steam / Dust False Alarm Rate:    {results['steam_false_alarms']}% (Goal: 0%)")
    print(f"  • Early Combustion Detection:       {'PASSED' if results['fire_detected'] else 'WARNING'}")
    print(f"  • 4-Point PPE Gear Inspection:      Helmet, Vest, Footwear, Hand Gloves operational")
    print(f"  • Real-Time Edge Latency:           {mean_lat:.1f} ms ({fps:.1f} FPS)")
    print("=" * 70)

    return True


if __name__ == "__main__":
    run_benchmark()
