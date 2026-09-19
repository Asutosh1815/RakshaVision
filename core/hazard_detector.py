"""
Hazard Detector module for RakshaVision.
Detects early signs of smoke and fire/flame in real-time CCTV or recorded video feeds
using multi-spectral chromatic analysis (HSV + YCrCb + RGB gradient) and spatial contour clustering.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2

from core.types import BoundingBox, HazardDetection, HazardType, SeverityLevel


class HazardDetector:
    def __init__(
        self,
        min_fire_area: int = 250,
        min_smoke_area: int = 600,
        sensitivity: float = 0.5
    ):
        self.min_fire_area = min_fire_area
        self.min_smoke_area = min_smoke_area
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.prev_gray: Optional[np.ndarray] = None

    def detect_hazards(
        self,
        frame: np.ndarray,
        worker_boxes: Optional[List[BoundingBox]] = None
    ) -> List[HazardDetection]:
        """
        Processes a video frame and returns all detected fire and smoke hazards
        with bounding boxes, confidence ratings, and severity levels.
        worker_boxes: If provided, worker regions are isolated to prevent orange vests from false fire triggering.
        """
        hazards: List[HazardDetection] = []
        h, w, _ = frame.shape
        total_frame_area = float(h * w)

        # 1. Fire Detection Pipeline
        fire_detections = self._detect_flames(frame, total_frame_area, worker_boxes=worker_boxes)
        hazards.extend(fire_detections)

        # 2. Smoke Detection Pipeline
        smoke_detections = self._detect_smoke(frame, total_frame_area, worker_boxes=worker_boxes)
        hazards.extend(smoke_detections)

        return hazards

    def _detect_flames(
        self,
        frame: np.ndarray,
        total_frame_area: float,
        worker_boxes: Optional[List[BoundingBox]] = None
    ) -> List[HazardDetection]:
        detections: List[HazardDetection] = []
        h, w, _ = frame.shape

        # Convert colorspaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        b, g, r = cv2.split(frame)
        y, cr, cb = cv2.split(ycrcb)

        # Rule 1: RGB chromatic condition for fire (Red significantly exceeds Green, which exceeds Blue)
        r_thresh = int(180 - (self.sensitivity * 30))
        # Fire is predominantly red/orange: R must be clearly greater than G, and G strictly greater than B
        rgb_fire = (r.astype(np.int32) - g.astype(np.int32) > 22) & (g > b) & (r > r_thresh)

        # Rule 2: HSV color model for fire (Bright red, orange, warm red-yellow)
        # Exclude pure fluorescent lime/yellow paint (Hue > 24)
        lower_fire1 = np.array([0, int(115 - self.sensitivity * 30), 180], dtype=np.uint8)
        upper_fire1 = np.array([22, 255, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv, lower_fire1, upper_fire1)

        lower_fire2 = np.array([172, int(115 - self.sensitivity * 30), 180], dtype=np.uint8)
        upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # Rule 3: YCrCb rule for active combustion: Cr strictly greater than Cb by at least 28
        ycrcb_fire = (y > cb) & (cr.astype(np.int32) - cb.astype(np.int32) > 28) & (cr > 148)

        # Combine rules
        combined = (rgb_fire & ycrcb_fire & (mask_hsv > 0)).astype(np.uint8) * 255

        # CRITICAL FIX: Mask out detected workers so orange safety vests NEVER trigger fire alerts
        if worker_boxes:
            worker_mask = np.ones((h, w), dtype=np.uint8) * 255
            for wb in worker_boxes:
                mx = int(0.12 * wb.width)
                my = int(0.06 * wb.height)
                wx1 = max(0, int(wb.x1 - mx))
                wy1 = max(0, int(wb.y1 - my))
                wx2 = min(w, int(wb.x2 + mx))
                wy2 = min(h, int(wb.y2 + my))
                # Zero out worker body
                worker_mask[wy1:wy2, wx1:wx2] = 0
            combined = cv2.bitwise_and(combined, worker_mask)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=3)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        raw_confs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and ignore whole-frame lighting artifacts
            if area >= self.min_fire_area and area < (total_frame_area * 0.70):
                x, y_box, bw, bh = cv2.boundingRect(cnt)

                # Aspect ratio check & density check
                roi = combined[y_box:y_box + bh, x:x + bw]
                density = cv2.countNonZero(roi) / float(bw * bh)

                # Active flame texture test (spatial intensity variance)
                gray_roi = cv2.cvtColor(frame[y_box:y_box + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
                std_dev = np.std(gray_roi)

                if density > 0.22 and std_dev > 15.0:
                    conf = min(0.99, 0.72 + (area / 10000.0) * 0.15 + (density * 0.12))
                    raw_boxes.append([x, y_box, x + bw, y_box + bh])
                    raw_confs.append(float(conf))

        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confs, score_threshold=0.5, nms_threshold=0.3)
            for i in indices:
                idx = int(i[0] if isinstance(i, (list, tuple, np.ndarray)) else i)
                bx = raw_boxes[idx]
                detections.append(HazardDetection(
                    hazard_type=HazardType.FIRE,
                    confidence=float(round(raw_confs[idx], 2)),
                    box=BoundingBox(float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                    severity=SeverityLevel.CRITICAL,
                    description="Active fire / open flame detected"
                ))
        return detections

    def _detect_smoke(
        self,
        frame: np.ndarray,
        total_frame_area: float,
        worker_boxes: Optional[List[BoundingBox]] = None
    ) -> List[HazardDetection]:
        detections: List[HazardDetection] = []
        h, w, _ = frame.shape

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # White / Grayish Smoke: low saturation, moderate-to-high brightness
        max_s = int(38 + (self.sensitivity * 12))
        min_v = 115
        max_v = 232
        smoke_mask = (s < max_s) & (v >= min_v) & (v <= max_v)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Smoke has diffuse boundaries - filter out harsh sharp edges
        soft_boundary = np.abs(laplacian) < 22.0

        smoke_combined = (smoke_mask & soft_boundary).astype(np.uint8) * 255

        if worker_boxes:
            w_mask = np.ones((h, w), dtype=np.uint8) * 255
            for wb in worker_boxes:
                mx = int(0.15 * wb.width)
                my = int(0.08 * wb.height)
                wx1 = max(0, int(wb.x1 - mx))
                wy1 = max(0, int(wb.y1 - my))
                wx2 = min(w, int(wb.x2 + mx))
                wy2 = min(h, int(wb.y2 + my))
                w_mask[wy1:wy2, wx1:wx2] = 0
            smoke_combined = cv2.bitwise_and(smoke_combined, w_mask)

        # Morphological smoothing to identify continuous smoke plumes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        smoke_combined = cv2.morphologyEx(smoke_combined, cv2.MORPH_OPEN, kernel)
        smoke_combined = cv2.dilate(smoke_combined, kernel, iterations=3)

        contours, _ = cv2.findContours(smoke_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_s_boxes = []
        raw_s_confs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Smoke plumes are airborne and have significant spatial extent
            if area >= max(4800, self.min_smoke_area) and area < (total_frame_area * 0.38):
                x, y_box, bw, bh = cv2.boundingRect(cnt)

                # Filter out tiny or thin fragments
                if bw < 75 or bh < 75:
                    continue
                
                # Exclude ground floor plane (smoke rises upward)
                if (y_box + bh) > (h * 0.90) and y_box > (h * 0.60):
                    continue

                # Aspect ratio check: smoke clouds are volumetric, not thin lines
                ar = float(bw) / float(bh) if bh > 0 else 0
                if ar < 0.35 or ar > 2.8:
                    continue

                roi = smoke_combined[y_box:y_box + bh, x:x + bw]
                density = cv2.countNonZero(roi) / float(bw * bh)

                # Check diffuse plume standard deviation (14.0 <= std <= 45.0)
                # Rejects cluttered pipes/machinery (std > 50) and flat walls (std < 12)
                gray_roi = gray[y_box:y_box + bh, x:x + bw]
                std_val = float(np.std(gray_roi))

                if density > 0.28 and (14.0 <= std_val <= 46.0):
                    conf = min(0.95, 0.62 + (area / 15000.0) * 0.2 + (density * 0.15))
                    raw_s_boxes.append([x, y_box, x + bw, y_box + bh])
                    raw_s_confs.append(float(conf))

        if raw_s_boxes:
            indices = cv2.dnn.NMSBoxes(raw_s_boxes, raw_s_confs, score_threshold=0.5, nms_threshold=0.3)
            for i in indices:
                idx = int(i[0] if isinstance(i, (list, tuple, np.ndarray)) else i)
                bx = raw_s_boxes[idx]
                detections.append(HazardDetection(
                    hazard_type=HazardType.SMOKE,
                    confidence=float(round(raw_s_confs[idx], 2)),
                    box=BoundingBox(float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                    severity=SeverityLevel.CRITICAL if (bx[2]-bx[0])*(bx[3]-bx[1]) > 5000 else SeverityLevel.HIGH,
                    description="Early smoke plume detected"
                ))
        return detections
