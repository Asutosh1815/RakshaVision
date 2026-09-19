"""
Incident Logger and Dispatch module for RakshaVision.
Logs safety infractions and hazardous environmental triggers, generates alert payloads,
maintains audit history with snapshot crops, and exports CSV reports.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import cv2
import numpy as np

from core.types import (
    SafetyIncident,
    WorkerCompliance,
    HazardDetection,
    ComplianceStatus,
    HazardType,
    SeverityLevel,
    ZoneConfig
)


class IncidentLogger:
    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.incidents: List[SafetyIncident] = []
        self._last_alert_time: Dict[str, float] = {}
        self.alert_cooldown_seconds: float = 4.0  # Cooldown between repeat alerts for same object/worker

    def log_ppe_violations(
        self,
        frame: np.ndarray,
        workers: List[WorkerCompliance],
        zone: ZoneConfig
    ) -> List[SafetyIncident]:
        """Logs newly detected worker violations with cooldown deduplication."""
        new_incidents: List[SafetyIncident] = []
        now = time.time()

        for w in workers:
            if w.status in (ComplianceStatus.VIOLATION, ComplianceStatus.WARNING):
                key = f"{zone.zone_id}_worker_{w.worker_id}"
                last_time = self._last_alert_time.get(key, 0.0)

                if now - last_time >= self.alert_cooldown_seconds:
                    self._last_alert_time[key] = now
                    
                    missing_str = ", ".join(w.missing_items) if w.missing_items else "Incomplete PPE"
                    severity = SeverityLevel.HIGH if ("Helmet" in w.missing_items or "Safety Vest" in w.missing_items) else SeverityLevel.MEDIUM
                    
                    # Save snapshot thumbnail
                    snapshot_path = self._save_worker_snapshot(frame, w, zone.zone_id)

                    incident = SafetyIncident(
                        incident_id=f"INC-PPE-{int(now * 1000) % 1000000:06d}",
                        timestamp=now,
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        camera_id=zone.camera_id,
                        incident_type="PPE_VIOLATION",
                        severity=severity,
                        details=f"Worker #{w.worker_id} violation: Missing [{missing_str}]",
                        snapshot_path=snapshot_path,
                        acknowledged=False
                    )
                    self.incidents.append(incident)
                    new_incidents.append(incident)

        return new_incidents

    def log_hazard_incidents(
        self,
        frame: np.ndarray,
        hazards: List[HazardDetection],
        zone: ZoneConfig
    ) -> List[SafetyIncident]:
        """Logs fire or smoke hazards with instant priority."""
        new_incidents: List[SafetyIncident] = []
        now = time.time()

        for h in hazards:
            key = f"{zone.zone_id}_{h.hazard_type.value}"
            last_time = self._last_alert_time.get(key, 0.0)

            if now - last_time >= (self.alert_cooldown_seconds * 0.75):
                self._last_alert_time[key] = now
                snapshot_path = self._save_hazard_snapshot(frame, h, zone.zone_id)

                inc_type = "FIRE_HAZARD" if h.hazard_type == HazardType.FIRE else "SMOKE_HAZARD"

                incident = SafetyIncident(
                    incident_id=f"INC-{h.hazard_type.value[:4]}-{int(now * 1000) % 1000000:06d}",
                    timestamp=now,
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    camera_id=zone.camera_id,
                    incident_type=inc_type,
                    severity=h.severity,
                    details=f"{h.description} (Confidence: {int(h.confidence * 100)}%) in {zone.location_desc}",
                    snapshot_path=snapshot_path,
                    acknowledged=False
                )
                self.incidents.append(incident)
                new_incidents.append(incident)

        return new_incidents

    def _save_worker_snapshot(self, frame: np.ndarray, worker: WorkerCompliance, zone_id: str) -> Optional[str]:
        try:
            h, w, _ = frame.shape
            x1, y1, x2, y2 = worker.box.to_int_tuple()
            # Add margin
            mx = int(0.08 * worker.box.width)
            my = int(0.08 * worker.box.height)
            sx1 = max(0, x1 - mx)
            sy1 = max(0, y1 - my)
            sx2 = min(w, x2 + mx)
            sy2 = min(h, y2 + my)

            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                fname = f"snap_ppe_{zone_id}_w{worker.worker_id}_{int(time.time()*1000)}.jpg"
                fpath = os.path.join(self.snapshot_dir, fname)
                cv2.imwrite(fpath, crop)
                return fpath
        except Exception:
            pass
        return None

    def _save_hazard_snapshot(self, frame: np.ndarray, hazard: HazardDetection, zone_id: str) -> Optional[str]:
        try:
            h, w, _ = frame.shape
            x1, y1, x2, y2 = hazard.box.to_int_tuple()
            mx = int(0.15 * hazard.box.width)
            my = int(0.15 * hazard.box.height)
            sx1 = max(0, x1 - mx)
            sy1 = max(0, y1 - my)
            sx2 = min(w, x2 + mx)
            sy2 = min(h, y2 + my)

            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                fname = f"snap_hazard_{hazard.hazard_type.value.lower()}_{zone_id}_{int(time.time()*1000)}.jpg"
                fpath = os.path.join(self.snapshot_dir, fname)
                cv2.imwrite(fpath, crop)
                return fpath
        except Exception:
            pass
        return None

    def acknowledge_incident(self, incident_id: str) -> bool:
        for inc in self.incidents:
            if inc.incident_id == incident_id:
                inc.acknowledged = True
                return True
        return False

    def get_recent_incidents(self, limit: int = 50) -> List[SafetyIncident]:
        return sorted(self.incidents, key=lambda x: x.timestamp, reverse=True)[:limit]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.incidents:
            return pd.DataFrame(columns=[
                "Incident ID", "Time", "Zone", "Camera", "Type", "Severity", "Details", "Status"
            ])

        records = []
        for inc in reversed(self.incidents):
            records.append({
                "Incident ID": inc.incident_id,
                "Time": datetime.fromtimestamp(inc.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "Zone": inc.zone_name,
                "Camera": inc.camera_id,
                "Type": inc.incident_type,
                "Severity": inc.severity.value,
                "Details": inc.details,
                "Status": "RESOLVED / ACK" if inc.acknowledged else "ACTIVE ALERT"
            })
        return pd.DataFrame(records)

    def export_csv(self, file_path: str = "safety_audit_log.csv") -> str:
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)
        return file_path
