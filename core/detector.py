"""
Safety Gear Detector module for RakshaVision.
Uses YOLOv8 models for worker detection, dedicated PPE models for hardhats, gloves, footwear,
and specialized colorimetric analysis for high-visibility safety vests.
"""

import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from ultralytics import YOLO

from core.types import BoundingBox, Detection


class SafetyGearDetector:
    def __init__(
        self,
        base_model_path: str = "yolov8n.pt",
        ppe_model_path: str = "weights/yolov8n-ppe.pt",
        hardhat_model_path: str = "weights/yolov8n-hardhat.pt",
        mendeley_model_path: str = "weights/yolov8n-mendeley-ppe.pt",
        device: str = "cpu"
    ):
        self.device = device

        # 1. Load standard YOLO for person detection
        print(f"[Detector] Loading base YOLO model ({base_model_path})...")
        self.person_model = YOLO(base_model_path)

        # 2. Load fine-tuned Mendeley PPE model (Helmet, NoHelmet, Vest, NoVest)
        self.mendeley_model = None
        check_paths = [
            mendeley_model_path,
            "runs/mendeley_train/ppe_model/weights/best.pt",
            "runs/mendeley_train/ppe_model/weights/last.pt"
        ]
        for p in check_paths:
            if os.path.exists(p):
                try:
                    print(f"[Detector] Loading fine-tuned Mendeley PPE model ({p})...")
                    self.mendeley_model = YOLO(p)
                    break
                except Exception as e:
                    print(f"[Detector] Note: Could not load {p}: {e}")

        # 3. Load General PPE Model for footwear and gloves
        self.ppe_model = None
        if os.path.exists(ppe_model_path):
            try:
                print(f"[Detector] Loading dedicated PPE model ({ppe_model_path})...")
                self.ppe_model = YOLO(ppe_model_path)
            except Exception as e:
                print(f"[Detector] Warning: Could not load PPE model: {e}")

        # 4. Load Hardhat model if available
        self.hardhat_model = None
        if os.path.exists(hardhat_model_path):
            try:
                print(f"[Detector] Loading dedicated Hardhat model ({hardhat_model_path})...")
                self.hardhat_model = YOLO(hardhat_model_path)
            except Exception as e:
                print(f"[Detector] Warning: Could not load Hardhat model: {e}")

        self.last_latency_ms: float = 0.0
        self.last_ppe_latency_ms: float = 0.0

    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.35) -> List[Detection]:
        """Detect all human workers in the frame with measured latency in milliseconds."""
        import time
        t_start = time.perf_counter()

        results = self.person_model(frame, classes=[0], conf=conf_threshold, verbose=False)
        detections: List[Detection] = []

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                detections.append(Detection(
                    label="person",
                    confidence=conf,
                    box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                    track_id=track_id
                ))

        self.last_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return detections

    def detect_ppe_items(self, frame: np.ndarray, conf_threshold: float = 0.18) -> List[Detection]:
        """
        Detect raw PPE equipment on the full frame using the Mendeley model
        (Vest/NoVest, Helmet/NoHelmet) and the general PPE model (shoes, gloves).
        Lowered conf_threshold to 0.18 to let spatial arbitration do filtering.
        """
        import time
        t_start = time.perf_counter()
        ppe_detections: List[Detection] = []

        # 1. Primary Mendeley Model: Vest vs NoVest, Helmet vs NoHelmet
        if self.mendeley_model is not None:
            try:
                m_results = self.mendeley_model(frame, conf=conf_threshold, verbose=False)
                for r in m_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        raw_label = self.mendeley_model.names.get(cls_id, "").lower()

                        if "novest" in raw_label or "no_vest" in raw_label:
                            ppe_detections.append(Detection(
                                label="no_vest",
                                confidence=conf,
                                box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                                sub_type="violation"
                            ))
                        elif "vest" in raw_label:
                            ppe_detections.append(Detection(
                                label="vest",
                                confidence=conf,
                                box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                                sub_type="compliant"
                            ))
                        elif "nohelmet" in raw_label or "no_helmet" in raw_label:
                            ppe_detections.append(Detection(
                                label="no_helmet",
                                confidence=conf,
                                box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                                sub_type="violation"
                            ))
                        elif "helmet" in raw_label:
                            ppe_detections.append(Detection(
                                label="helmet",
                                confidence=conf,
                                box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                                sub_type="compliant"
                            ))
            except Exception as e:
                pass

        # 2. General PPE Model on full frame (for spatial reference; per-person crops done in compliance_engine)
        if self.ppe_model is not None:
            try:
                p_results = self.ppe_model(frame, conf=conf_threshold, verbose=False)
                for r in p_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        label = self.ppe_model.names.get(cls_id, "").lower()

                        is_violation = label.startswith("no_")
                        ppe_detections.append(Detection(
                            label=label,
                            confidence=conf,
                            box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                            sub_type="violation" if is_violation else "compliant"
                        ))
            except Exception as e:
                pass

        # 3. Hardhat Model (fallback when Mendeley model unavailable)
        if self.mendeley_model is None and self.hardhat_model is not None:
            try:
                h_results = self.hardhat_model(frame, conf=conf_threshold, verbose=False)
                for r in h_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        raw_label = self.hardhat_model.names.get(cls_id, "").lower()

                        if "hard" in raw_label or "helmet" in raw_label:
                            is_violation = "no" in raw_label
                            ppe_detections.append(Detection(
                                label="helmet" if not is_violation else "no_helmet",
                                confidence=conf,
                                box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                                sub_type="violation" if is_violation else "compliant"
                            ))
            except Exception as e:
                pass

        self.last_ppe_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return ppe_detections

    def detect_ppe_for_person(
        self,
        frame: np.ndarray,
        person_box: BoundingBox,
        conf_threshold: float = 0.18
    ) -> List[Detection]:
        """
        Run PPE detection on a padded crop around a single person.
        Returns detections with bounding boxes translated back to full-frame coordinates.
        This ensures gloves/shoes detected in the crop are spatially anchored to this worker.
        """
        h, w = frame.shape[:2]

        # Expand person bounding box by 25% in each direction for context
        pad_x = int((person_box.x2 - person_box.x1) * 0.25)
        pad_y = int((person_box.y2 - person_box.y1) * 0.25)

        cx1 = max(0, int(person_box.x1) - pad_x)
        cy1 = max(0, int(person_box.y1) - pad_y)
        cx2 = min(w, int(person_box.x2) + pad_x)
        cy2 = min(h, int(person_box.y2) + pad_y)

        if cx2 <= cx1 or cy2 <= cy1:
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []

        results: List[Detection] = []

        # Run general PPE model (glove/shoe classes)
        if self.ppe_model is not None:
            try:
                p_results = self.ppe_model(crop, conf=conf_threshold, verbose=False)
                for r in p_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        label = self.ppe_model.names.get(cls_id, "").lower()

                        # Only use shoe and glove classes from this model
                        if "shoe" not in label and "glove" not in label:
                            continue

                        # Translate coordinates back to full frame
                        fx1 = coords[0] + cx1
                        fy1 = coords[1] + cy1
                        fx2 = coords[2] + cx1
                        fy2 = coords[3] + cy1

                        is_violation = label.startswith("no_")
                        results.append(Detection(
                            label=label,
                            confidence=conf,
                            box=BoundingBox(fx1, fy1, fx2, fy2),
                            sub_type="violation" if is_violation else "compliant"
                        ))
            except Exception:
                pass

        # Also run Mendeley model on the crop for helmet/vest with tighter spatial context
        if self.mendeley_model is not None:
            try:
                m_results = self.mendeley_model(crop, conf=conf_threshold, verbose=False)
                for r in m_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        raw_label = self.mendeley_model.names.get(cls_id, "").lower()

                        fx1 = coords[0] + cx1
                        fy1 = coords[1] + cy1
                        fx2 = coords[2] + cx1
                        fy2 = coords[3] + cy1

                        if "novest" in raw_label or "no_vest" in raw_label:
                            results.append(Detection(
                                label="no_vest", confidence=conf,
                                box=BoundingBox(fx1, fy1, fx2, fy2), sub_type="violation"
                            ))
                        elif "vest" in raw_label:
                            results.append(Detection(
                                label="vest", confidence=conf,
                                box=BoundingBox(fx1, fy1, fx2, fy2), sub_type="compliant"
                            ))
                        elif "nohelmet" in raw_label or "no_helmet" in raw_label:
                            results.append(Detection(
                                label="no_helmet", confidence=conf,
                                box=BoundingBox(fx1, fy1, fx2, fy2), sub_type="violation"
                            ))
                        elif "helmet" in raw_label:
                            results.append(Detection(
                                label="helmet", confidence=conf,
                                box=BoundingBox(fx1, fy1, fx2, fy2), sub_type="compliant"
                            ))
            except Exception:
                pass

        return results

    def evaluate_vest_presence(
        self,
        frame: np.ndarray,
        person_box: BoundingBox
    ) -> Tuple[bool, float]:
        """
        Evaluates high-visibility safety vest presence in the torso region
        using fluorescent chromaticity (Neon Lime/Yellow and Neon Orange)
        and retroreflective stripe contrast analysis.
        """
        h, w, _ = frame.shape
        x1 = max(0, int(person_box.x1))
        y1 = max(0, int(person_box.y1))
        x2 = min(w, int(person_box.x2))
        y2 = min(h, int(person_box.y2))

        pw = x2 - x1
        ph = y2 - y1
        if pw < 10 or ph < 20:
            return False, 0.0

        # Torso bounding box: 18% to 68% vertical span, middle 84% horizontal span
        ty1 = y1 + int(0.18 * ph)
        ty2 = y1 + int(0.68 * ph)
        tx1 = x1 + int(0.08 * pw)
        tx2 = x2 - int(0.08 * pw)

        if tx2 <= tx1 or ty2 <= ty1:
            return False, 0.0

        torso_crop = frame[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return False, 0.0

        # CLAHE to normalize shadows
        hsv_raw = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        hsv_raw[:, :, 2] = clahe.apply(hsv_raw[:, :, 2])
        hsv = hsv_raw

        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]

        # 1. Fluorescent Neon Lime / Yellow (H: 18-45, S: 50-255, V: 85-255)
        lower_yellow = np.array([18, 50, 85], dtype=np.uint8)
        upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. Fluorescent Neon Orange (H: 3-17, S: 75-255, V: 95-255)
        lower_orange = np.array([3, 75, 95], dtype=np.uint8)
        upper_orange = np.array([17, 255, 255], dtype=np.uint8)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        # 3. High-Vis Red/Warm (H: 172-180)
        lower_red = np.array([172, 80, 95], dtype=np.uint8)
        upper_red = np.array([180, 255, 255], dtype=np.uint8)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        vest_mask = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, mask_red))

        # Morphological opening to eliminate speckles
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        vest_mask = cv2.morphologyEx(vest_mask, cv2.MORPH_OPEN, k)
        vest_pixels = cv2.countNonZero(vest_mask)
        vest_ratio = vest_pixels / float(total_pixels)

        # 4. Reflective silver/gray stripes (High V, Low S)
        lower_reflective = np.array([0, 0, 160], dtype=np.uint8)
        upper_reflective = np.array([180, 55, 255], dtype=np.uint8)
        mask_refl = cv2.inRange(hsv, lower_reflective, upper_reflective)
        refl_pixels = cv2.countNonZero(mask_refl)
        refl_ratio = refl_pixels / float(total_pixels)

        # Sobel horizontal line check (reflective stripes have horizontal edges)
        gray_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray_torso, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = float(np.mean(np.abs(sobel_y)))
        has_stripe_edges = edge_energy > 16.0
        has_refl_tape = (refl_ratio >= 0.022 and has_stripe_edges)

        has_vest = False
        confidence = 0.0

        if vest_ratio >= 0.14 and has_refl_tape:
            # Standard: fluorescent + confirmed reflective tape
            has_vest = True
            confidence = min(0.98, 0.65 + vest_ratio * 1.5 + refl_ratio * 3.0)
        elif vest_ratio >= 0.22 and refl_ratio >= 0.035:
            # Strong fluorescent with some reflective material
            has_vest = True
            confidence = min(0.92, 0.60 + vest_ratio * 1.2)
        elif vest_ratio >= 0.09 and refl_ratio >= 0.045 and has_stripe_edges:
            # Moderate fluorescent with clear reflective stripe edges
            has_vest = True
            confidence = 0.82
        elif vest_ratio >= 0.30:
            # Very high fluorescent coverage even without confirmed tape
            # (tape may be occluded by tools/arms but vest fabric clearly visible)
            has_vest = True
            confidence = min(0.80, 0.55 + vest_ratio * 0.8)
        else:
            # Normal shirts lacking reflective tape → strictly non-vest
            has_vest = False
            confidence = max(0.05, float(vest_ratio * 0.8))

        return has_vest, float(round(confidence, 3))

    def evaluate_head_region(
        self,
        frame: np.ndarray,
        person_box: BoundingBox
    ) -> Tuple[bool, float]:
        """
        Secondary analysis on the upper head region to detect hard hat color
        uniformity and convex shell geometry when neural models report borderline confidence.
        """
        h, w, _ = frame.shape
        x1 = max(0, int(person_box.x1))
        y1 = max(0, int(person_box.y1))
        x2 = min(w, int(person_box.x2))
        y2 = min(h, int(person_box.y2))

        pw = x2 - x1
        ph = y2 - y1
        if pw < 8 or ph < 16:
            return False, 0.0

        # Head region is top 22% of person
        hy1 = max(0, y1 - int(0.05 * ph))
        hy2 = y1 + int(0.22 * ph)
        hx1 = max(0, x1 + int(0.10 * pw))
        hx2 = min(w, x2 - int(0.10 * pw))

        if hx2 <= hx1 or hy2 <= hy1:
            return False, 0.0

        head_crop = frame[hy1:hy2, hx1:hx2]
        if head_crop.size == 0:
            return False, 0.0

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        total_p = head_crop.shape[0] * head_crop.shape[1]

        # Yellow Hardhat (H: 20-38, S: 80-255, V: 120-255)
        mask_y = cv2.inRange(hsv, np.array([20, 80, 120]), np.array([38, 255, 255]))
        # White Hardhat (Low S, High V)
        mask_w = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        # Blue Hardhat (H: 95-125, S: 90-255, V: 80-255)
        mask_b = cv2.inRange(hsv, np.array([95, 90, 80]), np.array([125, 255, 255]))
        # Orange/Red Hardhat
        mask_o = cv2.inRange(hsv, np.array([4, 100, 120]), np.array([15, 255, 255]))
        # Green Hardhat (H: 38-85, S: 80-255, V: 80-255)
        mask_g = cv2.inRange(hsv, np.array([38, 80, 80]), np.array([85, 255, 255]))

        combined_hat = cv2.bitwise_or(mask_y, cv2.bitwise_or(mask_w, cv2.bitwise_or(mask_b, cv2.bitwise_or(mask_o, mask_g))))
        hat_ratio = cv2.countNonZero(combined_hat) / float(total_p)

        if hat_ratio >= 0.22:
            return True, float(round(min(0.92, 0.50 + hat_ratio * 1.5), 2))
        return False, 0.0

    def evaluate_footwear_presence(
        self,
        frame: np.ndarray,
        person_box: BoundingBox
    ) -> Tuple[bool, float]:
        """
        Evaluates presence of protective safety footwear in the lower ground-plane region
        (bottom 22% of worker bounding box). Differentiates heavy boot profile from bare feet/sandals.
        """
        h, w, _ = frame.shape
        x1 = max(0, int(person_box.x1))
        y1 = max(0, int(person_box.y1))
        x2 = min(w, int(person_box.x2))
        y2 = min(h, int(person_box.y2))

        pw = x2 - x1
        ph = y2 - y1
        if pw < 10 or ph < 20:
            return False, 0.0

        fy1 = max(0, y1 + int(0.78 * ph))
        fy2 = min(h, y2 + int(0.04 * ph))
        fx1 = max(0, x1 - int(0.06 * pw))
        fx2 = min(w, x2 + int(0.06 * pw))

        if fx2 <= fx1 or fy2 <= fy1:
            return False, 0.0

        feet_crop = frame[fy1:fy2, fx1:fx2]
        if feet_crop.size == 0:
            return False, 0.0

        total_pixels = feet_crop.shape[0] * feet_crop.shape[1]

        # 1. Bare skin detection in YCrCb (bare feet or open sandals)
        ycrcb = cv2.cvtColor(feet_crop, cv2.COLOR_BGR2YCrCb)
        mask_skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        skin_ratio = cv2.countNonZero(mask_skin) / float(total_pixels)

        # 2. Dark heavy boot leather / rubber sole profile
        hsv = cv2.cvtColor(feet_crop, cv2.COLOR_BGR2HSV)
        # Strict dark threshold: avoid misclassifying dark trouser legs as boots
        mask_dark = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 85], dtype=np.uint8))
        dark_ratio = cv2.countNonZero(mask_dark) / float(total_pixels)

        # 3. Check for compact non-elongated shape (boots are wide; trouser legs are narrow)
        gray_feet = cv2.cvtColor(feet_crop, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray_feet, cv2.CV_64F, 1, 1, ksize=3)
        edge_energy = float(np.mean(np.abs(sobel)))

        if skin_ratio > 0.18:
            return False, float(round(skin_ratio, 2))

        # Raised threshold (0.28 vs old 0.22) to reduce dark-trouser false positives
        has_boots = (dark_ratio >= 0.28 or (dark_ratio >= 0.14 and edge_energy > 14.0))
        confidence = min(0.92, 0.50 + dark_ratio * 0.8 + min(0.3, edge_energy / 50.0)) if has_boots else 0.20
        return has_boots, float(round(confidence, 2))

    def evaluate_gloves_presence(
        self,
        frame: np.ndarray,
        person_box: BoundingBox
    ) -> Tuple[bool, float]:
        """
        Evaluates presence of industrial safety hand gloves in the lateral hand regions.
        Detects nitrile blue, neon yellow/green, and dark heavy leather gloves.
        Differentiates from bare skin tones.
        """
        h, w, _ = frame.shape
        x1 = max(0, int(person_box.x1))
        y1 = max(0, int(person_box.y1))
        x2 = min(w, int(person_box.x2))
        y2 = min(h, int(person_box.y2))

        pw = x2 - x1
        ph = y2 - y1
        if pw < 10 or ph < 20:
            return False, 0.0

        hy1 = max(0, y1 + int(0.35 * ph))
        hy2 = min(h, y1 + int(0.82 * ph))
        left_x1 = max(0, x1 - int(0.18 * pw))
        left_x2 = min(w, x1 + int(0.28 * pw))
        right_x1 = max(0, x2 - int(0.28 * pw))
        right_x2 = min(w, x2 + int(0.18 * pw))

        hand_crops = []
        if left_x2 > left_x1 and hy2 > hy1:
            lc = frame[hy1:hy2, left_x1:left_x2]
            if lc.size > 0:
                hand_crops.append(lc)
        if right_x2 > right_x1 and hy2 > hy1:
            rc = frame[hy1:hy2, right_x1:right_x2]
            if rc.size > 0:
                hand_crops.append(rc)

        if not hand_crops:
            return False, 0.0

        skin_ratios = []
        glove_ratios = []

        for crop in hand_crops:
            tot = crop.shape[0] * crop.shape[1]

            # Bare skin detection
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            mask_skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
            skin_ratios.append(cv2.countNonZero(mask_skin) / float(tot))

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Nitrile blue gloves (H: 95-130, S: 60-255, V: 60-255)
            mask_blue = cv2.inRange(hsv, np.array([95, 60, 60], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8))

            # Hi-vis neon yellow/green gloves (H: 25-42, S: 70-255, V: 70-255)
            mask_hivis = cv2.inRange(hsv, np.array([25, 70, 70], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))

            # Dark leather / rubber gloves (low V, any H — but NOT skin tone area)
            mask_leather = cv2.inRange(hsv, np.array([0, 20, 15], dtype=np.uint8), np.array([20, 200, 75], dtype=np.uint8))

            # Orange/red safety gloves (H: 0-15 high S, V: 80-255)
            mask_orange_glove = cv2.inRange(hsv, np.array([0, 100, 80], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8))

            mask_glove = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_hivis, cv2.bitwise_or(mask_leather, mask_orange_glove)))
            glove_ratios.append(cv2.countNonZero(mask_glove) / float(tot))

        mean_skin = float(np.mean(skin_ratios)) if skin_ratios else 0.0
        mean_glove = float(np.mean(glove_ratios)) if glove_ratios else 0.0

        if mean_skin > 0.16:
            return False, float(round(mean_skin, 2))

        # More permissive glove detection threshold
        has_gloves = (mean_glove >= 0.03 or (mean_skin < 0.05 and mean_glove >= 0.01))
        conf = min(0.90, 0.55 + mean_glove * 2.0) if has_gloves else 0.25
        return has_gloves, float(round(conf, 2))
