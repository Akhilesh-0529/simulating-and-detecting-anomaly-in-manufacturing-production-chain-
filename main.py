#!/usr/bin/env python3
"""
Smart Factory Monitoring System — Main Entry Point
====================================================
Cognizant Technoverse Hackathon Submission

Demonstrates:
  • Real-time manufacturing defect detection (YOLOv8 / simulation mode)
  • Worker PPE compliance checking via object detection
  • LLM-based root-cause analysis for each detected anomaly
  • Structured JSON reporting + rich ANSI console output

Usage:
    python main.py

Requirements: Python 3.8+ (stdlib only). Install `ultralytics` for
real YOLOv8 inference; falls back to high-fidelity simulation otherwise.
"""

import sys
import time
import os

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from simulation.production_chain import ProductionChainSimulator
from detection.yolo_detector import SmartFactoryYOLODetector
from analysis.root_cause_llm import LLMRootCauseAnalyzer
from reporting.alert_reporter import FactoryAlertReporter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATIONS = [
    "STATION_1 — CNC Milling",
    "STATION_2 — Thermal Treatment",
    "STATION_3 — Robotic Assembly",
    "STATION_4 — QA Worker Zone",
]

STATION_IDS = ["STATION_1", "STATION_2", "STATION_3", "STATION_4"]

CYCLES = [
    # (cycle_label, anomaly_injections: list of (anomaly_type, station_id))
    ("CYCLE_01_NORMAL",    []),
    ("CYCLE_02_VIBRATION", [("spindle_wear_vibration", "STATION_1")]),
    ("CYCLE_03_DUAL",      [("thermal_furnace_drift", "STATION_2"),
                            ("worker_ppe_violation", "STATION_4")]),
    ("CYCLE_04_PNEUMATIC", [("pneumatic_pressure_drop", "STATION_3")]),
    ("CYCLE_05_RECOVERY",  []),
]


# ---------------------------------------------------------------------------
# Helper: convert Detection objects → plain dicts for reporter
# ---------------------------------------------------------------------------
def _det_to_dict(det) -> dict:
    bbox = []
    if det.bbox:
        bbox = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
    return {
        "class":      det.class_name,
        "confidence": round(det.confidence, 3),
        "bbox":       bbox,
        "category":   det.category,
    }


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------
def main() -> None:
    reporter = FactoryAlertReporter(output_dir="outputs")
    reporter.print_header()

    simulator = ProductionChainSimulator(seed=42)
    detector  = SmartFactoryYOLODetector(conf_threshold=0.5)
    analyzer  = LLMRootCauseAnalyzer(use_ollama=True)

    # Print detection engine mode
    engine_mode = "YOLOv8 (ultralytics)" if detector.model is not None else "Synthetic YOLOv8 Simulation"
    print(f"  Detection Engine : {engine_mode}")
    print(f"  LLM Engine       : {'Ollama' if analyzer._check_ollama() else 'Template Engine (Ollama not running)'}")
    print(f"  Stations Active  : {len(STATIONS)}")
    print()

    for cycle_num, (batch_label, anomalies) in enumerate(CYCLES, start=1):

        # 1. Inject scenario anomalies into simulator
        for anomaly_type, station_id in anomalies:
            simulator.inject_anomaly(anomaly_type, station_id, duration_steps=3)

        # 2. Generate inspection targets for this cycle
        targets = simulator.generate_inspection_cycle(batch_id=cycle_num)

        # 3. Collect telemetry for each station
        telemetry_map: dict = {}
        for sid, sname in zip(STATION_IDS, STATIONS):
            snap = simulator.get_telemetry(sid)
            telemetry_map[sname] = {
                "temperature_c":    round(snap.temperature, 1),
                "vibration_mm_s":   round(snap.vibration, 2),
                "pressure_bar":     round(snap.pressure, 2),
                "conveyor_speed":   round(snap.conveyor_speed, 2),
                "status":           snap.status,
            }

        # 4. Run YOLOv8 detection on each target
        detections_map: dict = {s: [] for s in STATIONS}
        for target in targets:
            dets = detector.detect_target(target)
            station_name = STATIONS[STATION_IDS.index(target.station_id)]
            detections_map[station_name].extend([_det_to_dict(d) for d in dets])

        # 5. LLM root-cause analysis for each station with detections
        root_causes_map: dict = {}
        for sname in STATIONS:
            dets = detections_map[sname]
            if dets:
                defect_types = list({d["class"] for d in dets})
                tel = telemetry_map[sname]
                root_causes_map[sname] = analyzer.analyze_batch(
                    defect_types=defect_types,
                    station=sname,
                    telemetry=tel,
                )
            else:
                root_causes_map[sname] = []

        # 6. Report
        reporter.report_cycle(
            cycle_id=cycle_num,
            batch_id=batch_label,
            stations=STATIONS,
            detections_map=detections_map,
            telemetry_map=telemetry_map,
            root_causes_map=root_causes_map,
        )

        # Simulate real-time pacing (brief delay between cycles)
        if cycle_num < len(CYCLES):
            time.sleep(0.4)

    # 7. Summary
    reporter.print_summary(total_cycles=len(CYCLES))


if __name__ == "__main__":
    main()
