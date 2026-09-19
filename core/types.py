"""
Data models and type definitions for RakshaVision Industrial Safety Monitoring.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import time


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"


class HazardType(str, Enum):
    FIRE = "FIRE"
    SMOKE = "SMOKE"
    NONE = "NONE"


class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"    # Fire, dense smoke, dangerous multiple violations
    HIGH = "HIGH"            # Missing mandatory hardhat or safety vest
    MEDIUM = "MEDIUM"        # Missing gloves or boots in mandatory zones
    LOW = "LOW"              # Minor warning / informational


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def contains_point(self, px: float, py: float) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


@dataclass
class Detection:
    label: str
    confidence: float
    box: BoundingBox
    track_id: Optional[int] = None
    sub_type: Optional[str] = None  # e.g., 'compliant' or 'missing'


@dataclass
class WorkerCompliance:
    worker_id: int
    box: BoundingBox
    status: ComplianceStatus
    has_helmet: bool = False
    has_vest: bool = False
    has_boots: bool = False
    has_gloves: bool = False
    helmet_conf: float = 0.0
    vest_conf: float = 0.0
    boots_conf: float = 0.0
    gloves_conf: float = 0.0
    missing_items: List[str] = field(default_factory=list)
    active_items: List[str] = field(default_factory=list)


@dataclass
class HazardDetection:
    hazard_type: HazardType
    confidence: float
    box: BoundingBox
    severity: SeverityLevel
    description: str


@dataclass
class SafetyIncident:
    incident_id: str
    timestamp: float
    zone_id: str
    zone_name: str
    camera_id: str
    incident_type: str        # 'PPE_VIOLATION', 'FIRE_HAZARD', 'SMOKE_HAZARD'
    severity: SeverityLevel
    details: str
    snapshot_path: Optional[str] = None
    acknowledged: bool = False


@dataclass
class ZoneConfig:
    zone_id: str
    name: str
    camera_id: str
    location_desc: str
    require_helmet: bool = True
    require_vest: bool = True
    require_boots: bool = False
    require_gloves: bool = False
    hazard_monitoring: bool = True
    sensitivity: float = 0.4
