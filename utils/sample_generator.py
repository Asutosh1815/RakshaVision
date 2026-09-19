"""
Sample Asset Generator and Manager for RakshaVision.
Generates realistic sample CCTV scenes, workers with/without PPE, and hazard scenarios
to ensure instant demonstration and automated testability out of the box.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple


class SampleAssetManager:
    def __init__(self, asset_dir: str = "demo_assets"):
        self.asset_dir = asset_dir
        os.makedirs(self.asset_dir, exist_ok=True)

    def prepare_all_samples(self):
        """Generates all standard demo scenarios if not present."""
        self.create_compliant_worker_scene()
        self.create_violation_worker_scene()
        self.create_fire_hazard_scene()
        self.create_multi_worker_factory_video()

    def create_compliant_worker_scene(self) -> str:
        path = os.path.join(self.asset_dir, "scenario_compliant.jpg")
        if os.path.exists(path):
            return path

        img = self._create_factory_backdrop(width=1280, height=720)
        # Worker 1: Center-left, wearing yellow helmet, neon yellow high-vis vest
        self._draw_synthetic_worker(
            img, center_x=450, base_y=580, scale=1.4,
            has_helmet=True, helmet_color=(0, 215, 255),  # Yellow
            has_vest=True, vest_color=(0, 240, 200)      # High-vis neon lime/yellow
        )
        # Worker 2: Center-right, wearing white helmet, neon orange vest
        self._draw_synthetic_worker(
            img, center_x=850, base_y=550, scale=1.3,
            has_helmet=True, helmet_color=(240, 240, 240), # White
            has_vest=True, vest_color=(0, 120, 255)       # Neon Orange
        )
        cv2.imwrite(path, img)
        return path

    def create_violation_worker_scene(self) -> str:
        path = os.path.join(self.asset_dir, "scenario_violation.jpg")
        if os.path.exists(path):
            return path

        img = self._create_factory_backdrop(width=1280, height=720)
        # Worker 1: Missing both Helmet and Vest (Violation)
        self._draw_synthetic_worker(
            img, center_x=400, base_y=570, scale=1.35,
            has_helmet=False,
            has_vest=False,
            shirt_color=(80, 50, 40) # Plain dark brown jacket
        )
        # Worker 2: Wearing Helmet, but NO Vest (Violation)
        self._draw_synthetic_worker(
            img, center_x=820, base_y=580, scale=1.4,
            has_helmet=True, helmet_color=(0, 215, 255),
            has_vest=False,
            shirt_color=(40, 40, 110) # Plain dark blue jacket
        )
        cv2.imwrite(path, img)
        return path

    def create_fire_hazard_scene(self) -> str:
        path = os.path.join(self.asset_dir, "scenario_fire_hazard.jpg")
        if os.path.exists(path):
            return path

        img = self._create_factory_backdrop(width=1280, height=720)
        # Worker running away from hazard area
        self._draw_synthetic_worker(
            img, center_x=320, base_y=590, scale=1.4,
            has_helmet=True, helmet_color=(0, 215, 255),
            has_vest=True, vest_color=(0, 240, 200)
        )
        # Draw active Fire Flame in right corner
        self._draw_fire_flame(img, center_x=960, base_y=520, width=160, height=220)
        # Draw Smoke Plume above flame
        self._draw_smoke_plume(img, center_x=980, base_y=280, radius=90)

        cv2.imwrite(path, img)
        return path

    def create_multi_worker_factory_video(self, num_frames: int = 120) -> str:
        path = os.path.join(self.asset_dir, "cctv_simulation.mp4")
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            return path

        w, h = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 25.0, (w, h))

        for f in range(num_frames):
            frame = self._create_factory_backdrop(width=w, height=h)
            
            # Worker 1 (Compliant): Walking from left to right
            w1_x = int(250 + (f * 4.5))
            self._draw_synthetic_worker(
                frame, center_x=w1_x, base_y=580, scale=1.35,
                has_helmet=True, helmet_color=(0, 215, 255),
                has_vest=True, vest_color=(0, 240, 200)
            )

            # Worker 2 (Violating): Stationary near CNC machine without helmet
            self._draw_synthetic_worker(
                frame, center_x=900, base_y=560, scale=1.3,
                has_helmet=False,
                has_vest=False,
                shirt_color=(50, 50, 90)
            )

            # Fire hazard starts developing around frame 50
            if f >= 45:
                flame_intensity = min(1.0, (f - 45) / 35.0)
                fh = int(180 * flame_intensity)
                fw = int(140 * flame_intensity)
                if fh > 20:
                    self._draw_fire_flame(frame, center_x=650, base_y=490, width=fw, height=fh)
                    self._draw_smoke_plume(frame, center_x=660, base_y=280, radius=int(75 * flame_intensity))

            writer.write(frame)

        writer.release()
        return path

    def _create_factory_backdrop(self, width: int = 1280, height: int = 720) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient industrial wall
        for y in range(height):
            v = int(45 + (y / float(height)) * 40)
            img[y, :] = (v, v, v + 8)

        # Industrial concrete floor
        floor_y = int(height * 0.62)
        cv2.rectangle(img, (0, floor_y), (width, height), (70, 75, 80), -1)

        # Safety yellow/black hazard warning floor line
        stripe_w = 40
        for x in range(0, width, stripe_w * 2):
            pts = np.array([
                [x, floor_y + 10], [x + stripe_w, floor_y + 10],
                [x + stripe_w - 20, floor_y + 30], [x - 20, floor_y + 30]
            ], np.int32)
            cv2.fillPoly(img, [pts], (0, 215, 255))

        # Background machinery and structural pillars
        cv2.rectangle(img, (80, 80), (220, floor_y), (35, 38, 42), -1)
        cv2.rectangle(img, (1050, 60), (1220, floor_y), (35, 38, 42), -1)
        cv2.putText(img, "BAY 02: CNC & HEAVY FABRICATION", (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 130, 140), 2)

        return img

    def _draw_synthetic_worker(
        self,
        img: np.ndarray,
        center_x: int,
        base_y: int,
        scale: float = 1.0,
        has_helmet: bool = True,
        helmet_color: Tuple[int, int, int] = (0, 215, 255),
        has_vest: bool = True,
        vest_color: Tuple[int, int, int] = (0, 240, 200),
        shirt_color: Tuple[int, int, int] = (60, 60, 70)
    ):
        h = int(260 * scale)
        w = int(80 * scale)
        top_y = base_y - h

        # Head / Hair or Helmet
        head_radius = int(22 * scale)
        head_center = (center_x, top_y + head_radius + 5)
        
        # Draw Face / Skin
        cv2.circle(img, head_center, head_radius, (170, 195, 235), -1)

        if has_helmet:
            # Draw Hard Hat dome over head
            helmet_pts = np.array([
                [center_x - int(head_radius * 1.3), head_center[1]],
                [center_x - int(head_radius * 1.2), head_center[1] - int(head_radius * 1.2)],
                [center_x, head_center[1] - int(head_radius * 1.6)],
                [center_x + int(head_radius * 1.2), head_center[1] - int(head_radius * 1.2)],
                [center_x + int(head_radius * 1.3), head_center[1]],
            ], np.int32)
            cv2.fillPoly(img, [helmet_pts], helmet_color)
            # Helmet brim
            cv2.ellipse(img, (center_x, head_center[1]), (int(head_radius * 1.4), int(head_radius * 0.4)), 0, 0, 360, helmet_color, -1)
        else:
            # Natural dark hair
            cv2.ellipse(img, (center_x, head_center[1] - int(head_radius * 0.4)), (head_radius, int(head_radius * 0.7)), 0, 180, 360, (20, 20, 25), -1)

        # Torso
        torso_top = head_center[1] + head_radius
        torso_bot = torso_top + int(90 * scale)
        torso_w = int(w * 0.95)
        tx1 = center_x - torso_w // 2
        tx2 = center_x + torso_w // 2

        # Base shirt
        cv2.rectangle(img, (tx1, torso_top), (tx2, torso_bot), shirt_color, -1)

        # Vest (if wearing)
        if has_vest:
            cv2.rectangle(img, (tx1, torso_top), (tx2, torso_bot), vest_color, -1)
            # Silver reflective stripes across vest
            stripe_y1 = torso_top + int(30 * scale)
            stripe_y2 = torso_top + int(60 * scale)
            cv2.rectangle(img, (tx1, stripe_y1), (tx2, stripe_y1 + int(8 * scale)), (240, 240, 245), -1)
            cv2.rectangle(img, (tx1, stripe_y2), (tx2, stripe_y2 + int(8 * scale)), (240, 240, 245), -1)

        # Legs / Pants
        leg_bot = base_y
        pant_color = (60, 65, 80)
        leg_w = int(w * 0.35)
        # Left leg
        cv2.rectangle(img, (center_x - torso_w // 2, torso_bot), (center_x - 3, leg_bot), pant_color, -1)
        # Right leg
        cv2.rectangle(img, (center_x + 3, torso_bot), (center_x + torso_w // 2, leg_bot), pant_color, -1)

        # Safety Boots
        cv2.rectangle(img, (center_x - torso_w // 2 - 4, leg_bot - 14), (center_x - 1, leg_bot), (20, 20, 20), -1)
        cv2.rectangle(img, (center_x + 1, leg_bot - 14), (center_x + torso_w // 2 + 4, leg_bot), (20, 20, 20), -1)

    def _draw_fire_flame(self, img: np.ndarray, center_x: int, base_y: int, width: int, height: int):
        """Draws realistic chromatic flame cluster."""
        # Outer flame (Bright red-orange)
        pts_outer = np.array([
            [center_x - width // 2, base_y],
            [center_x - width // 3, base_y - height // 2],
            [center_x - width // 5, base_y - int(height * 0.85)],
            [center_x, base_y - height],
            [center_x + width // 5, base_y - int(height * 0.75)],
            [center_x + width // 3, base_y - height // 2],
            [center_x + width // 2, base_y]
        ], np.int32)
        cv2.fillPoly(img, [pts_outer], (0, 69, 255)) # BGR Flame Red

        # Inner flame (Bright yellow core)
        pts_inner = np.array([
            [center_x - width // 4, base_y],
            [center_x - width // 6, base_y - height // 3],
            [center_x, base_y - int(height * 0.65)],
            [center_x + width // 6, base_y - height // 3],
            [center_x + width // 4, base_y]
        ], np.int32)
        cv2.fillPoly(img, [pts_inner], (0, 230, 255)) # BGR Yellow Flame

    def _draw_smoke_plume(self, img: np.ndarray, center_x: int, base_y: int, radius: int):
        """Draws volumetric smoke clouds."""
        overlay = img.copy()
        for i in range(5):
            ox = center_x + np.random.randint(-radius // 2, radius // 2)
            oy = base_y + np.random.randint(-radius // 2, radius // 2)
            r = int(radius * (0.6 + i * 0.15))
            val = int(140 + i * 15)
            cv2.circle(overlay, (ox, oy), r, (val, val, val), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
