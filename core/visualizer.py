"""
Visualizer module for RakshaVision.
Renders real-time CCTV heads-up display (HUD), color-coded worker compliance bounding boxes,
detailed PPE checklist badges, and high-visibility emergency hazard alerts.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
import time

from core.types import (
    BoundingBox,
    WorkerCompliance,
    HazardDetection,
    ComplianceStatus,
    HazardType,
    ZoneConfig
)


class SafetyVisualizer:
    # Color palette (BGR format for OpenCV)
    COLOR_SAFE = (50, 205, 50)       # Emerald Lime Green
    COLOR_DANGER = (30, 30, 235)     # Bright Crimson Red
    COLOR_WARN = (0, 165, 255)       # Amber Orange
    COLOR_HUD_BG = (24, 24, 28)      # Deep Gunmetal
    COLOR_CYAN = (235, 206, 0)       # Tech Cyan
    COLOR_WHITE = (255, 255, 255)

    def draw_hud(
        self,
        frame: np.ndarray,
        workers: List[WorkerCompliance],
        hazards: List[HazardDetection],
        zone: ZoneConfig,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        show_boxes: bool = True,
        show_badges: bool = True,
        show_hazards: bool = True,
        show_telemetry: bool = True,
        show_confidence: bool = False
    ) -> np.ndarray:
        """
        Draws the complete safety monitoring telemetry HUD onto the frame
        with customizable layer toggles and measured end-to-end latency.
        """
        vis_frame = frame.copy()

        # 1. Draw Hazards (Fire & Smoke) First
        if show_hazards and hazards:
            self._draw_hazards(vis_frame, hazards)

        # 2. Draw Worker Compliance Bounding Boxes & Badges
        if (show_boxes or show_badges) and workers:
            self._draw_workers(vis_frame, workers, zone, show_boxes, show_badges, show_confidence)

        # 3. Draw Top Information & Telemetry Banner
        if show_telemetry:
            self._draw_top_telemetry(vis_frame, workers, hazards, zone, fps, latency_ms)

        # 4. If critical hazard exists, draw perimeter alert strobe
        if show_hazards and any(h.hazard_type in (HazardType.FIRE, HazardType.SMOKE) for h in hazards):
            self._draw_perimeter_strobe(vis_frame)

        return vis_frame

    def _draw_workers(
        self,
        frame: np.ndarray,
        workers: List[WorkerCompliance],
        zone: ZoneConfig,
        show_boxes: bool = True,
        show_badges: bool = True,
        show_confidence: bool = True
    ):
        for worker in workers:
            x1, y1, x2, y2 = worker.box.to_int_tuple()
            pw = x2 - x1
            ph = y2 - y1

            # Determine color
            if worker.status == ComplianceStatus.COMPLIANT:
                color = self.COLOR_SAFE
                status_text = "COMPLIANT"
            elif worker.status == ComplianceStatus.VIOLATION:
                color = self.COLOR_DANGER
                status_text = "VIOLATION"
            elif worker.status == ComplianceStatus.WARNING:
                color = self.COLOR_WARN
                status_text = "CAUTION"
            else:
                color = self.COLOR_CYAN
                status_text = "UNKNOWN"

            if show_boxes:
                # Draw corner brackets for high-tech HUD look
                self._draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=min(25, int(pw * 0.25)))

            if show_badges:
                # Main header label background
                header_str = f"W#{worker.worker_id}: {status_text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.48
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(header_str, font, font_scale, thickness)

                badge_y1 = max(0, y1 - text_h - 10)
                badge_y2 = y1
                cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 14, badge_y2), color, -1)
                cv2.putText(frame, header_str, (x1 + 6, badge_y2 - 5), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, header_str, (x1 + 6, badge_y2 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

                # Detailed PPE checklist tag underneath
                tag_y = y1 + 18
                gear_items = []
                if zone.require_helmet:
                    gear_items.append(("Helmet", worker.has_helmet, worker.helmet_conf))
                if zone.require_vest:
                    gear_items.append(("Vest", worker.has_vest, worker.vest_conf))
                if zone.require_boots:
                    gear_items.append(("Boots", worker.has_boots, worker.boots_conf))
                if zone.require_gloves:
                    gear_items.append(("Gloves", worker.has_gloves, worker.gloves_conf))

                for gear_name, present, conf in gear_items:
                    indicator = "[+]" if present else "[X]"
                    item_str = f"{indicator} {gear_name}"
                    item_col = self.COLOR_SAFE if present else self.COLOR_DANGER

                    (iw, ih), _ = cv2.getTextSize(item_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                    # Semi-transparent text background
                    cv2.rectangle(frame, (x1 + 4, tag_y - 12), (x1 + max(95, iw + 10), tag_y + 4), (20, 20, 24), -1)
                    cv2.putText(frame, item_str, (x1 + 6, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, item_col, 1, cv2.LINE_AA)
                    tag_y += 19

    def _draw_hazards(self, frame: np.ndarray, hazards: List[HazardDetection]):
        for h in hazards:
            x1, y1, x2, y2 = h.box.to_int_tuple()

            if h.hazard_type == HazardType.FIRE:
                color = (0, 69, 255)  # Bright Flame Red-Orange
                label = "CRITICAL HAZARD: FIRE FLAME"
            else:
                color = (200, 200, 200) # Smoke Gray
                label = "HAZARD: SMOKE PLUME"

            # Draw dashed / thick hazard box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw hazard stripes corner
            self._draw_corner_box(frame, x1, y1, x2, y2, color, thickness=3, corner_len=30)

            # Header alert label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 16, y1), color, -1)
            cv2.putText(frame, label, (x1 + 8, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x1 + 8, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_top_telemetry(
        self,
        frame: np.ndarray,
        workers: List[WorkerCompliance],
        hazards: List[HazardDetection],
        zone: ZoneConfig,
        fps: float,
        latency_ms: float = 0.0
    ):
        h, w, _ = frame.shape
        bar_height = 54

        # Overlay dark semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Bottom border line of top bar
        has_fire = any(h.hazard_type == HazardType.FIRE for h in hazards)
        border_col = self.COLOR_DANGER if has_fire else self.COLOR_CYAN
        cv2.line(frame, (0, bar_height), (w, bar_height), border_col, 2)

        # Camera & Zone Text
        cam_text = f"RAKSHAVISION AI  |  {zone.camera_id}: {zone.name}"
        cv2.putText(frame, cam_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        time_str = time.strftime("%Y-%m-%d  %H:%M:%S")
        rec_text = f"REC [LIVE]  {time_str}  |  FPS: {fps:.1f}  |  LATENCY: {latency_ms:.1f}ms"
        cv2.putText(frame, rec_text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Right stats: Total Workers, Compliant, Violations
        total = len(workers)
        violations = sum(1 for w in workers if w.status == ComplianceStatus.VIOLATION)
        compliant = sum(1 for w in workers if w.status == ComplianceStatus.COMPLIANT)
        comp_rate = int(round((compliant / float(total)) * 100)) if total > 0 else 100

        # Compliance score pill
        rate_color = self.COLOR_SAFE if comp_rate >= 80 else (self.COLOR_WARN if comp_rate >= 50 else self.COLOR_DANGER)
        rate_str = f"SITE COMPLIANCE: {comp_rate}%"
        (rw, rh), _ = cv2.getTextSize(rate_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        rx = w - rw - 24
        cv2.rectangle(frame, (rx - 10, 10), (w - 14, 44), (40, 40, 48), -1)
        cv2.rectangle(frame, (rx - 10, 10), (w - 14, 44), rate_color, 2)
        cv2.putText(frame, rate_str, (rx, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rate_color, 2, cv2.LINE_AA)

        # Workers & Violations stats text
        stats_str = f"WORKERS: {total}   |   SAFE: {compliant}   |   VIOLATIONS: {violations}"
        (sw, sh), _ = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, stats_str, (rx - sw - 28, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_perimeter_strobe(self, frame: np.ndarray):
        """Draws emergency red hazard borders around the frame perimeter."""
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), self.COLOR_DANGER, 6)

    def _draw_corner_box(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 20
    ):
        # Full subtle outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
