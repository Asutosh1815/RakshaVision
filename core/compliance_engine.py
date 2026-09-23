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

            # 1. Helmet Evaluation with Confidence Arbitration
            helmet_confs = []
            no_helmet_confs = []
            for item in raw_ppe:
                if "helmet" in item.label or "hard" in item.label:
                    cx, cy = item.box.center
                    if head_region.contains_point(cx, cy) or head_region.iou(item.box) > 0.04:
                        if "no" not in item.label:
                            helmet_confs.append(item.confidence)
                        else:
                            no_helmet_confs.append(item.confidence)

            has_helmet = False
            helmet_conf = 0.0

            max_h = max(helmet_confs) if helmet_confs else 0.0
            max_nh = max(no_helmet_confs) if no_helmet_confs else 0.0

            if max_h > max_nh and max_h >= 0.28:
                has_helmet = True
                helmet_conf = float(round(max_h, 2))
            elif max_nh > max_h and max_nh >= 0.30:
                has_helmet = False
                helmet_conf = float(round(max_nh, 2))
            else:
                # Secondary head region geometric/color analysis
                sec_has_h, sec_conf = self.detector.evaluate_head_region(frame, pbox)
                if sec_has_h:
                    has_helmet = True
                    helmet_conf = sec_conf
                else:
                    has_helmet = False
                    helmet_conf = max(max_h, max_nh, 0.15)

            # 2. High-Visibility Vest Evaluation (Neural Model + Strict Optical Verification)
            vest_region = BoundingBox(
                pbox.x1 - 0.08 * pw,
                pbox.y1 + 0.14 * ph,
                pbox.x2 + 0.08 * pw,
                pbox.y1 + 0.72 * ph
            )
            vest_confs = []
            no_vest_confs = []
            for item in raw_ppe:
                if "vest" in item.label:
                    cx, cy = item.box.center
                    if vest_region.contains_point(cx, cy) or vest_region.iou(item.box) > 0.04:
                        if "no" not in item.label:
                            vest_confs.append(item.confidence)
                        else:
                            no_vest_confs.append(item.confidence)

            max_v = max(vest_confs) if vest_confs else 0.0
            max_nv = max(no_vest_confs) if no_vest_confs else 0.0

            has_vest = False
            vest_conf = 0.0

            if max_v > max_nv and max_v >= 0.28:
                has_vest = True
                vest_conf = float(round(max_v, 2))
            elif max_nv > max_v and max_nv >= 0.28:
                # Normal shirt detected by neural model - STRICTLY NOT A VEST
                has_vest = False
                vest_conf = float(round(max_nv, 2))
            else:
                # Secondary strict optical verification (requires certified retroreflective tape)
                sec_has_v, sec_v_conf = self.detector.evaluate_vest_presence(frame, pbox)
                has_vest = sec_has_v
                vest_conf = sec_v_conf if sec_has_v else max(max_v, max_nv, 0.10)

            # 3. Footwear (Boots/Shoes) with Spatial Ground Anchoring
            shoes_confs = []
            no_shoes_confs = []
            for item in raw_ppe:
                if "shoe" in item.label or "boot" in item.label:
                    cx, cy = item.box.center
                    if feet_region.contains_point(cx, cy) or feet_region.iou(item.box) > 0.04:
                        if "no" not in item.label:
                            shoes_confs.append(item.confidence)
                        else:
                            no_shoes_confs.append(item.confidence)

            max_s = max(shoes_confs) if shoes_confs else 0.0
            max_ns = max(no_shoes_confs) if no_shoes_confs else 0.0

            has_boots = False
            boots_conf = 0.0
            if max_s > max_ns and max_s >= 0.25:
                has_boots = True
                boots_conf = float(round(max_s, 2))
            elif max_ns > max_s and max_ns >= 0.28:
                has_boots = False
                boots_conf = float(round(max_ns, 2))
            else:
                # Secondary optical ground-plane inspection
                opt_boots, opt_b_conf = self.detector.evaluate_footwear_presence(frame, pbox)
                has_boots = opt_boots
                boots_conf = opt_b_conf

            # 4. Gloves Evaluation with Lateral Arm Anchoring
            glove_confs = []
            no_glove_confs = []
            for item in raw_ppe:
                if "glove" in item.label:
                    cx, cy = item.box.center
                    if hands_region.contains_point(cx, cy) or hands_region.iou(item.box) > 0.04:
                        if "no" not in item.label:
                            glove_confs.append(item.confidence)
                        else:
                            no_glove_confs.append(item.confidence)

            max_g = max(glove_confs) if glove_confs else 0.0
            max_ng = max(no_glove_confs) if no_glove_confs else 0.0

            has_gloves = False
            gloves_conf = 0.0
            if max_g > max_ng and max_g >= 0.25:
                has_gloves = True
                gloves_conf = float(round(max_g, 2))
            elif max_ng > max_g and max_ng >= 0.28:
                has_gloves = False
                gloves_conf = float(round(max_ng, 2))
            else:
                # Secondary optical hand-region skin vs. glove fabric analysis
                opt_gloves, opt_g_conf = self.detector.evaluate_gloves_presence(frame, pbox)
                has_gloves = opt_gloves
                gloves_conf = opt_g_conf

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
