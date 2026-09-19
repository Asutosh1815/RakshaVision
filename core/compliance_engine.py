"""
Compliance Engine for RakshaVision.
Performs anatomical spatial association between detected workers and PPE items,
enforces location-specific safety policies, and calculates real-time compliance metrics.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from core.types import (
    BoundingBox,
    Detection,
    WorkerCompliance,
    ComplianceStatus,
    ZoneConfig
)
from core.detector import SafetyGearDetector


class ComplianceEngine:
    def __init__(self, detector: SafetyGearDetector):
        self.detector = detector

    def evaluate_frame(
        self,
        frame: np.ndarray,
        persons: List[Detection],
        raw_ppe: List[Detection],
        zone: ZoneConfig
    ) -> List[WorkerCompliance]:
        """
        Associates detected PPE with each worker and evaluates compliance against the zone's policy.
        """
        worker_records: List[WorkerCompliance] = []

        for idx, person in enumerate(persons):
            worker_id = person.track_id if person.track_id is not None else (idx + 1)
            pbox = person.box
            ph = pbox.height
            pw = pbox.width

            # Define worker anatomical sub-regions
            head_region = BoundingBox(
                pbox.x1 - 0.05 * pw,
                pbox.y1 - 0.08 * ph,
                pbox.x2 + 0.05 * pw,
                pbox.y1 + 0.32 * ph
            )
            feet_region = BoundingBox(
                pbox.x1 - 0.10 * pw,
                pbox.y1 + 0.65 * ph,
                pbox.x2 + 0.10 * pw,
                pbox.y2 + 0.05 * ph
            )
            hands_region = BoundingBox(
                pbox.x1 - 0.20 * pw,
                pbox.y1 + 0.30 * ph,
                pbox.x2 + 0.20 * pw,
                pbox.y1 + 0.80 * ph
            )

            # 1. Helmet Evaluation
            has_helmet = False
            helmet_conf = 0.0
            for item in raw_ppe:
                if "helmet" in item.label or "hard" in item.label:
                    # Check if center of PPE item is inside worker's head region
                    cx, cy = item.box.center
                    if head_region.contains_point(cx, cy) or head_region.iou(item.box) > 0.05:
                        if item.label == "helmet" or item.label == "hardhat":
                            has_helmet = True
                            helmet_conf = max(helmet_conf, item.confidence)
                        elif item.label == "no_helmet" or item.label == "no-hardhat":
                            has_helmet = False
                            helmet_conf = max(helmet_conf, item.confidence)

            # 2. High-Visibility Vest Evaluation (Colorimetric & Reflective Analysis)
            has_vest, vest_conf = self.detector.evaluate_vest_presence(frame, pbox)

            # 3. Footwear (Boots/Shoes) Evaluation
            has_boots = False
            boots_conf = 0.0
            for item in raw_ppe:
                if "shoe" in item.label or "boot" in item.label:
                    cx, cy = item.box.center
                    if feet_region.contains_point(cx, cy) or feet_region.iou(item.box) > 0.05:
                        if item.label == "shoes" or item.label == "boot":
                            has_boots = True
                            boots_conf = max(boots_conf, item.confidence)
                        elif item.label == "no_shoes":
                            has_boots = False
                            boots_conf = max(boots_conf, item.confidence)

            # 4. Gloves Evaluation
            has_gloves = False
            gloves_conf = 0.0
            for item in raw_ppe:
                if "glove" in item.label:
                    cx, cy = item.box.center
                    if hands_region.contains_point(cx, cy) or hands_region.iou(item.box) > 0.05:
                        if item.label == "glove":
                            has_gloves = True
                            gloves_conf = max(gloves_conf, item.confidence)
                        elif item.label == "no_glove":
                            has_gloves = False
                            gloves_conf = max(gloves_conf, item.confidence)

            # 5. Evaluate Against Active Zone Policy
            missing_items = []
            active_items = []

            if zone.require_helmet:
                if not has_helmet:
                    missing_items.append("Helmet")
                else:
                    active_items.append("Helmet")
            elif has_helmet:
                active_items.append("Helmet")

            if zone.require_vest:
                if not has_vest:
                    missing_items.append("Safety Vest")
                else:
                    active_items.append("Safety Vest")
            elif has_vest:
                active_items.append("Safety Vest")

            if zone.require_boots:
                if not has_boots:
                    missing_items.append("Safety Boots")
                else:
                    active_items.append("Safety Boots")
            elif has_boots:
                active_items.append("Safety Boots")

            if zone.require_gloves:
                if not has_gloves:
                    missing_items.append("Gloves")
                else:
                    active_items.append("Gloves")
            elif has_gloves:
                active_items.append("Gloves")

            # Determine compliance status
            if len(missing_items) == 0:
                status = ComplianceStatus.COMPLIANT
            elif len(missing_items) == 1 and not (zone.require_helmet and "Helmet" in missing_items):
                status = ComplianceStatus.WARNING
            else:
                status = ComplianceStatus.VIOLATION

            worker_records.append(WorkerCompliance(
                worker_id=worker_id,
                box=pbox,
                status=status,
                has_helmet=has_helmet,
                has_vest=has_vest,
                has_boots=has_boots,
                has_gloves=has_gloves,
                helmet_conf=helmet_conf,
                vest_conf=vest_conf,
                boots_conf=boots_conf,
                gloves_conf=gloves_conf,
                missing_items=missing_items,
                active_items=active_items
            ))

        return worker_records

    @staticmethod
    def calculate_site_metrics(workers: List[WorkerCompliance]) -> Dict[str, float]:
        """Calculates summary KPIs across all workers in a frame."""
        total = len(workers)
        if total == 0:
            return {
                "total_workers": 0,
                "compliant_count": 0,
                "violation_count": 0,
                "warning_count": 0,
                "compliance_rate": 100.0
            }

        compliant = sum(1 for w in workers if w.status == ComplianceStatus.COMPLIANT)
        violations = sum(1 for w in workers if w.status == ComplianceStatus.VIOLATION)
        warnings = sum(1 for w in workers if w.status == ComplianceStatus.WARNING)
        rate = round((compliant / float(total)) * 100.0, 1)

        return {
            "total_workers": total,
            "compliant_count": compliant,
            "violation_count": violations,
            "warning_count": warnings,
            "compliance_rate": rate
        }
