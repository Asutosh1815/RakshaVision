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
        device: str = "cpu"
    ):
        self.device = device
        
        # Load standard YOLO for person detection
        print(f"[Detector] Loading base YOLO model ({base_model_path})...")
        self.person_model = YOLO(base_model_path)
        
        # Load PPE model if available
        self.ppe_model = None
        if os.path.exists(ppe_model_path):
            try:
                print(f"[Detector] Loading dedicated PPE model ({ppe_model_path})...")
                self.ppe_model = YOLO(ppe_model_path)
            except Exception as e:
                print(f"[Detector] Warning: Could not load PPE model: {e}")
                
        # Load Hardhat model if available
        self.hardhat_model = None
        if os.path.exists(hardhat_model_path):
            try:
                print(f"[Detector] Loading dedicated Hardhat model ({hardhat_model_path})...")
                self.hardhat_model = YOLO(hardhat_model_path)
            except Exception as e:
                print(f"[Detector] Warning: Could not load Hardhat model: {e}")

    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.35) -> List[Detection]:
        """Detect all human workers in the frame."""
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
        return detections

    def detect_ppe_items(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """Detect raw PPE equipment using YOLO PPE and Hardhat models."""
        ppe_detections: List[Detection] = []

        # 1. Hardhat Model (if available)
        if self.hardhat_model is not None:
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

        # 2. General PPE Model (if available)
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
                        clean_label = label.replace("no_", "")
                        
                        ppe_detections.append(Detection(
                            label=label,
                            confidence=conf,
                            box=BoundingBox(coords[0], coords[1], coords[2], coords[3]),
                            sub_type="violation" if is_violation else "compliant"
                        ))
            except Exception as e:
                pass

        return ppe_detections

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

        # Torso bounding box: 20% to 65% vertical span, middle 80% horizontal span
        ty1 = y1 + int(0.18 * ph)
        ty2 = y1 + int(0.65 * ph)
        tx1 = x1 + int(0.10 * pw)
        tx2 = x2 - int(0.10 * pw)

        if tx2 <= tx1 or ty2 <= ty1:
            return False, 0.0

        torso_crop = frame[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return False, 0.0

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]

        # 1. Fluorescent Neon Lime / Yellow (Hue: 18 - 42, Saturation: 60 - 255, Value: 100 - 255)
        lower_yellow = np.array([18, 60, 100], dtype=np.uint8)
        upper_yellow = np.array([42, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. Fluorescent Neon Orange / Red-Orange (Hue: 4 - 17, Saturation: 85 - 255, Value: 110 - 255)
        lower_orange = np.array([4, 85, 110], dtype=np.uint8)
        upper_orange = np.array([17, 255, 255], dtype=np.uint8)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        # Combined fluorescent high-vis mask
        vest_mask = cv2.bitwise_or(mask_yellow, mask_orange)
        vest_pixels = cv2.countNonZero(vest_mask)
        vest_ratio = vest_pixels / float(total_pixels)

        # 3. Reflective silver/white stripes (High V, Low S)
        lower_reflective = np.array([0, 0, 175], dtype=np.uint8)
        upper_reflective = np.array([180, 50, 255], dtype=np.uint8)
        mask_refl = cv2.inRange(hsv, lower_reflective, upper_reflective)
        refl_pixels = cv2.countNonZero(mask_refl)
        refl_ratio = refl_pixels / float(total_pixels)

        # High-vis vest score calculation
        # If fluorescent color ratio > 11% or (color ratio > 7% and reflective stripes present)
        confidence = 0.0
        has_vest = False

        if vest_ratio >= 0.12:
            confidence = min(0.98, 0.50 + vest_ratio * 1.5)
            has_vest = True
        elif vest_ratio >= 0.07 and refl_ratio >= 0.03:
            confidence = min(0.92, 0.45 + (vest_ratio + refl_ratio) * 1.4)
            has_vest = True
        elif vest_ratio >= 0.05:
            confidence = float(vest_ratio * 3.5)
            has_vest = False
        else:
            confidence = 0.10
            has_vest = False

        return has_vest, float(round(confidence, 3))
