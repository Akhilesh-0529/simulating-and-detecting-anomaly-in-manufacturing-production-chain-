"""
Production Chain Simulation Engine
Simulates multi-station manufacturing operations with real-time sensor telemetry
and visual inspection feeds.
"""

import time
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TelemetrySnapshot:
    station_id: str
    timestamp: float
    vibration: float     # mm/s
    temperature: float   # °C
    pressure: float      # bar
    conveyor_speed: float # m/min
    status: str = "NORMAL" # NORMAL, WARNING, CRITICAL
    anomaly_type: Optional[str] = None

@dataclass
class InspectionTarget:
    target_id: str
    target_type: str  # "part" or "worker"
    station_id: str
    timestamp: float
    ground_truth_anomalies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProductionChainSimulator:
    """
    Simulates a continuous manufacturing production line with telemetry streams
    and synthetic visual inspection frames.
    """
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.step_counter = 0
        self.active_anomalies: List[Dict[str, Any]] = []

    def inject_anomaly(self, anomaly_type: str, station_id: str, duration_steps: int = 5):
        """
        Inject an operational failure or safety violation into the production line.
        """
        self.active_anomalies.append({
            "type": anomaly_type,
            "station_id": station_id,
            "remaining_steps": duration_steps,
            "start_step": self.step_counter
        })

    def get_telemetry(self, station_id: str) -> TelemetrySnapshot:
        """
        Generate telemetry reading for a given station, reflecting any active anomalies.
        """
        now = time.time()
        # Baseline normal values with realistic noise
        vibration = random.gauss(2.0, 0.25)
        temperature = random.gauss(28.0, 1.2)
        pressure = random.gauss(6.1, 0.15)
        speed = random.gauss(1.2, 0.05)
        status = "NORMAL"
        detected_anomaly = None

        # Check for active injected anomalies
        for anom in self.active_anomalies:
            if anom["station_id"] == station_id and anom["remaining_steps"] > 0:
                atype = anom["type"]
                detected_anomaly = atype
                status = "CRITICAL"

                if atype == "spindle_wear_vibration":
                    vibration += random.uniform(3.5, 6.2) # High vibration spike
                elif atype == "thermal_furnace_drift":
                    temperature += random.uniform(45.0, 85.0) # Thermal runaway
                elif atype == "pneumatic_pressure_drop":
                    pressure -= random.uniform(2.5, 4.0) # Loss of pressure
                elif atype == "conveyor_belt_surge":
                    speed += random.uniform(0.8, 1.5)

        return TelemetrySnapshot(
            station_id=station_id,
            timestamp=now,
            vibration=round(max(0.1, vibration), 3),
            temperature=round(temperature, 2),
            pressure=round(max(0.2, pressure), 2),
            conveyor_speed=round(max(0.1, speed), 2),
            status=status,
            anomaly_type=detected_anomaly
        )

    def generate_inspection_cycle(self, batch_id: int) -> List[InspectionTarget]:
        """
        Generates simulated physical targets (manufactured parts & workers)
        passing through the cameras.
        """
        self.step_counter += 1
        targets = []
        now = time.time()

        # Update remaining steps for anomalies
        for anom in self.active_anomalies:
            anom["remaining_steps"] -= 1
        self.active_anomalies = [a for a in self.active_anomalies if a["remaining_steps"] > 0]

        # 1. Part on Station 1 (CNC Machining)
        p1_defects = []
        for anom in self.active_anomalies:
            if anom["station_id"] == "STATION_1" and anom["type"] == "spindle_wear_vibration":
                p1_defects.append("surface_scratch")
                if random.random() > 0.4:
                    p1_defects.append("hairline_crack")

        targets.append(InspectionTarget(
            target_id=f"PART-CNC-{batch_id:04d}",
            target_type="part",
            station_id="STATION_1",
            timestamp=now,
            ground_truth_anomalies=p1_defects,
            metadata={"part_name": "Aerospace Turbine Bracket", "material": "Ti-6Al-4V Titanium"}
        ))

        # 2. Part on Station 2 (Thermal Treatment)
        p2_defects = []
        for anom in self.active_anomalies:
            if anom["station_id"] == "STATION_2" and anom["type"] == "thermal_furnace_drift":
                p2_defects.append("dimensional_warp")
                p2_defects.append("hairline_crack")

        targets.append(InspectionTarget(
            target_id=f"PART-THM-{batch_id:04d}",
            target_type="part",
            station_id="STATION_2",
            timestamp=now,
            ground_truth_anomalies=p2_defects,
            metadata={"part_name": "Structural Flange", "material": "Alloy 7075"}
        ))

        # 3. Part on Station 3 (Robotic Assembly)
        p3_defects = []
        for anom in self.active_anomalies:
            if anom["station_id"] == "STATION_3" and anom["type"] == "pneumatic_pressure_drop":
                p3_defects.append("component_misalignment")

        targets.append(InspectionTarget(
            target_id=f"PART-ASM-{batch_id:04d}",
            target_type="part",
            station_id="STATION_3",
            timestamp=now,
            ground_truth_anomalies=p3_defects,
            metadata={"part_name": "Actuator Housing", "material": "Stainless Steel 316"}
        ))

        # 4. Worker on Station 4 (QA & Packaging Area)
        worker_violations = []
        for anom in self.active_anomalies:
            if anom["station_id"] == "STATION_4" and anom["type"] == "worker_ppe_violation":
                worker_violations.append("missing_helmet")
                worker_violations.append("missing_vest")

        targets.append(InspectionTarget(
            target_id=f"WORKER-QA-{random.randint(101, 108)}",
            target_type="worker",
            station_id="STATION_4",
            timestamp=now,
            ground_truth_anomalies=worker_violations,
            metadata={"zone": "High-Risk Packaging Conveyor", "shift": "Morning"}
        ))

        return targets
