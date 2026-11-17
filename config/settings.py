"""
Smart Factory Monitoring System - Configuration Settings
Cognizant Technoverse Hackathon Prototype
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Production Line Stations
STATIONS = {
    "STATION_1": {
        "id": "STATION_1",
        "name": "Precision CNC Milling & Machining",
        "camera_feed": "CAM_01_CNC",
        "primary_check": "dimensional_accuracy_and_surface",
        "normal_vibration_range": (1.0, 3.5),  # mm/s
        "normal_temp_range": (20.0, 38.0),     # °C
        "normal_pressure_range": (5.8, 6.4),   # bar
    },
    "STATION_2": {
        "id": "STATION_2",
        "name": "Thermal Treatment & Anodizing",
        "camera_feed": "CAM_02_THERMAL",
        "primary_check": "crack_detection_and_warping",
        "normal_vibration_range": (0.5, 2.0),
        "normal_temp_range": (180.0, 220.0),
        "normal_pressure_range": (2.0, 3.0),
    },
    "STATION_3": {
        "id": "STATION_3",
        "name": "Robotic Component Assembly",
        "camera_feed": "CAM_03_ASSEMBLY",
        "primary_check": "alignment_and_fasteners",
        "normal_vibration_range": (1.2, 4.0),
        "normal_temp_range": (22.0, 32.0),
        "normal_pressure_range": (6.0, 7.5),
    },
    "STATION_4": {
        "id": "STATION_4",
        "name": "Manual Final QA & Packaging",
        "camera_feed": "CAM_04_QA_WORKER",
        "primary_check": "ppe_compliance_and_packaging",
        "normal_vibration_range": (0.2, 1.5),
        "normal_temp_range": (20.0, 26.0),
        "normal_pressure_range": (4.5, 5.5),
    },
}

# YOLOv8 Defect Detection Classes
DEFECT_CLASSES = [
    "surface_scratch",
    "hairline_crack",
    "dimensional_warp",
    "weld_porosity",
    "component_misalignment",
    "dent_anomaly"
]

# YOLOv8 PPE (Safety) Classes
PPE_CLASSES = {
    "compliant": ["safety_helmet", "high_vis_vest", "protective_goggles", "safety_gloves"],
    "violations": ["missing_helmet", "missing_vest", "missing_goggles", "missing_gloves"]
}

# Root-Cause Severity Thresholds
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hybrid")  # options: "hybrid", "ollama", "openai", "gemini"
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
