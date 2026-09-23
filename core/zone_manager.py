"""
Zone Manager module for RakshaVision.
Provides location-specific factory context, multi-camera mapping, and zone safety policies.
"""

from typing import Dict, List, Optional
from core.types import ZoneConfig


class ZoneManager:
    def __init__(self):
        self.zones: Dict[str, ZoneConfig] = {
            "ZONE_FAB_BAY": ZoneConfig(
                zone_id="ZONE_FAB_BAY",
                name="Bay 1: Heavy Fabrication & CNC Machining",
                camera_id="CAM-01-FAB",
                location_desc="North Wing - Heavy Machinery & Overhead Crane Area",
                require_helmet=True,
                require_vest=True,
                require_boots=True,
                require_gloves=True,
                hazard_monitoring=True,
                sensitivity=0.45
            ),
            "ZONE_WELD_CHEM": ZoneConfig(
                zone_id="ZONE_WELD_CHEM",
                name="Bay 2: Welding, Cutting & Chemical Storage",
                camera_id="CAM-02-HOTWORK",
                location_desc="East Wing - Hot Work Permits, Flammable Solvents",
                require_helmet=True,
                require_vest=True,
                require_boots=True,
                require_gloves=True,
                hazard_monitoring=True,
                sensitivity=0.60
            ),
            "ZONE_WAREHOUSE": ZoneConfig(
                zone_id="ZONE_WAREHOUSE",
                name="Bay 3: Logistics, Loading Docks & Warehouse",
                camera_id="CAM-03-LOGISTICS",
                location_desc="South Yard - Active Forklift Traffic & High Pallet Racks",
                require_helmet=True,
                require_vest=True,
                require_boots=True,
                require_gloves=True,
                hazard_monitoring=True,
                sensitivity=0.40
            ),
            "ZONE_CONTROL_ROOM": ZoneConfig(
                zone_id="ZONE_CONTROL_ROOM",
                name="Zone 4: Central Control & Inspection Bay",
                camera_id="CAM-04-INSPECTION",
                location_desc="Administrative Core - Safety Transition Checkpoint",
                require_helmet=True,
                require_vest=False,
                require_boots=True,
                require_gloves=True,
                hazard_monitoring=False,
                sensitivity=0.30
            )
        }

    def get_all_zones(self) -> List[ZoneConfig]:
        return list(self.zones.values())

    def get_zone(self, zone_id: str) -> Optional[ZoneConfig]:
        return self.zones.get(zone_id)

    def update_zone_policy(
        self,
        zone_id: str,
        require_helmet: bool,
        require_vest: bool,
        require_boots: bool,
        require_gloves: bool,
        hazard_monitoring: bool,
        sensitivity: float
    ):
        if zone_id in self.zones:
            z = self.zones[zone_id]
            z.require_helmet = require_helmet
            z.require_vest = require_vest
            z.require_boots = require_boots
            z.require_gloves = require_gloves
            z.hazard_monitoring = hazard_monitoring
            z.sensitivity = sensitivity
