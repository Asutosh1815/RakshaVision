"""
Alert Router & Role-Based Dispatch Module for RakshaVision.
Enriches safety alerts with location context (Camera ID, Zone, Coordinates),
event classification, severity, target personnel roles, and actionable "What to Do Next" SOP protocols.
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.types import HazardType, SeverityLevel, ComplianceStatus, ZoneConfig


class PersonnelRole(str, Enum):
    FIRE_MARSHAL = "Fire Marshal & Emergency Response Team"
    SAFETY_OFFICER = "Chief Safety Officer & EHS Lead"
    FLOOR_SUPERVISOR = "Area Operations Supervisor"
    EQUIPMENT_FOREMAN = "Machinery & Equipment Foreman"
    MEDICAL_RESPONDER = "First Aid & Medical Station"


@dataclass
class ActionableAlert:
    alert_id: str
    timestamp: str
    camera_id: str
    zone_id: str
    zone_name: str
    location_desc: str
    event_type: str
    severity: SeverityLevel
    target_role: PersonnelRole
    target_channel: str
    action_protocol: List[str]
    context_summary: str
    confidence: float
    snapshot_path: Optional[str] = None
    acknowledged: bool = False


class AlertRouter:
    """
    Intelligent dispatch router mapping incidents to specific factory personnel
    with location metadata and step-by-step emergency SOPs.
    """

    # Actionable Standard Operating Procedures (SOPs)
    PROTOCOLS = {
        "FIRE_FLAME": [
            "1. Activate local pull station and trigger Bay automated sprinkler / CO2 deluge system.",
            "2. Evacuate all personnel via primary designated exit route immediately.",
            "3. Notify Emergency Response Team (ERT) and local Municipal Fire Department.",
            "4. Isolate electrical busbar and shut off flammable gas/solvent supply lines in the zone."
        ],
        "SMOKE_PLUME": [
            "1. Switch ventilation HVAC to Emergency Smoke Extraction / Purge Mode.",
            "2. Area supervisor to conduct immediate visual sweep with thermal imaging / gas detector.",
            "3. Halt hot work, welding, and grinding operations in adjacent bays.",
            "4. Verify muster station head count for the affected sector."
        ],
        "PPE_HELMET_MISSING": [
            "1. Floor supervisor to immediately issue stop-work order to non-compliant worker.",
            "2. Escort worker to the safety equipment kiosk for ANSI Z89.1 Type I/II hardhat issuance.",
            "3. Inspect overhead crane / suspended hoist status in the zone before work resumes.",
            "4. Record safety infraction in company EHS digital ledger."
        ],
        "PPE_VEST_MISSING": [
            "1. Prohibit worker movement within active mobile plant / forklift transit corridors.",
            "2. Provide certified EN ISO 20471 / ANSI Class 2 high-visibility safety vest.",
            "3. Notify Area Foreman to verify retroreflective striping under current ambient lighting."
        ],
        "PPE_BOOTS_MISSING": [
            "1. Restrict worker access to heavy fabrication, stamping, or drop-hazard areas.",
            "2. Mandate steel-toe / composite puncture-resistant safety footwear before re-entry."
        ],
        "PPE_GLOVES_MISSING": [
            "1. Halt handling of sharp sheet metal, abrasive parts, chemical fluids, or thermal tools.",
            "2. Issue task-specific cut-resistant (EN 388) or chemical-rated protective gloves."
        ]
    }

    def __init__(self):
        self.dispatched_alerts: List[ActionableAlert] = []

    def route_hazard(
        self,
        hazard_type: HazardType,
        confidence: float,
        zone: ZoneConfig,
        snapshot_path: Optional[str] = None
    ) -> ActionableAlert:
        """Route critical combustion hazards to emergency fire response teams."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_id = f"ALR-HAZ-{datetime.now().strftime('%H%M%S%f')[:10]}"

        if hazard_type == HazardType.FIRE:
            event_key = "FIRE_FLAME"
            severity = SeverityLevel.CRITICAL
            role = PersonnelRole.FIRE_MARSHAL
            channel = "RED-ALERT-RADIO & SIREN DELUGE (BROADCAST)"
            summary = f"Active combustion flame identified in {zone.name} ({zone.camera_id}). High urgency."
        else:
            event_key = "SMOKE_PLUME"
            severity = SeverityLevel.HIGH
            role = PersonnelRole.SAFETY_OFFICER
            channel = "EHS-MONITOR & HVAC CONTROLLER"
            summary = f"Dense particulate smoke plume spreading in {zone.name}. Possible smoldering combustion."

        alert = ActionableAlert(
            alert_id=alert_id,
            timestamp=now_str,
            camera_id=zone.camera_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            location_desc=zone.location_desc,
            event_type=f"HAZARD: {hazard_type.value}",
            severity=severity,
            target_role=role,
            target_channel=channel,
            action_protocol=self.PROTOCOLS.get(event_key, ["1. Investigate and secure the zone immediately."]),
            context_summary=summary,
            confidence=float(round(confidence, 2)),
            snapshot_path=snapshot_path
        )
        self.dispatched_alerts.insert(0, alert)
        return alert

    def route_ppe_violation(
        self,
        worker_id: int,
        missing_items: List[str],
        zone: ZoneConfig,
        confidence: float = 0.88,
        snapshot_path: Optional[str] = None
    ) -> ActionableAlert:
        """Route PPE compliance infractions to area floor supervisors and safety leads."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_id = f"ALR-PPE-{datetime.now().strftime('%H%M%S%f')[:10]}"

        protocols = []
        for item in missing_items:
            key = f"PPE_{item.upper().replace(' ', '_')}_MISSING"
            if "HELMET" in key:
                protocols.extend(self.PROTOCOLS["PPE_HELMET_MISSING"])
            elif "VEST" in key:
                protocols.extend(self.PROTOCOLS["PPE_VEST_MISSING"])
            elif "BOOT" in key or "FOOT" in key:
                protocols.extend(self.PROTOCOLS["PPE_BOOTS_MISSING"])
            elif "GLOVE" in key:
                protocols.extend(self.PROTOCOLS["PPE_GLOVES_MISSING"])

        if not protocols:
            protocols = ["1. Notify worker to wear required PPE before continuing operations."]

        severity = SeverityLevel.HIGH if any(x in ["Helmet", "Safety Vest"] for x in missing_items) else SeverityLevel.MEDIUM
        summary = f"Worker #{worker_id} in {zone.name} violating mandated gear: {', '.join(missing_items)}."

        alert = ActionableAlert(
            alert_id=alert_id,
            timestamp=now_str,
            camera_id=zone.camera_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            location_desc=zone.location_desc,
            event_type=f"PPE NON-COMPLIANCE: Worker #{worker_id}",
            severity=severity,
            target_role=PersonnelRole.FLOOR_SUPERVISOR,
            target_channel="SUPERVISOR PAGER / TABLET HUD",
            action_protocol=protocols,
            context_summary=summary,
            confidence=float(round(confidence, 2)),
            snapshot_path=snapshot_path
        )
        self.dispatched_alerts.insert(0, alert)
        return alert

    def get_recent_alerts(self, limit: int = 15) -> List[ActionableAlert]:
        return self.dispatched_alerts[:limit]

    def acknowledge_alert(self, alert_id: str):
        for a in self.dispatched_alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                break
