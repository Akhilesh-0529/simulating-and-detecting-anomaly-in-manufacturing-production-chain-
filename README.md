# 🏭 Smart Factory Monitoring System

> **Cognizant Technoverse Hackathon Submission**
> Real-time defect and PPE detection using **YOLOv8** object detection, combined with **LLM-based root-cause analysis**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)
![LLM](https://img.shields.io/badge/LLM-Root--Cause%20Analysis-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

The **Smart Factory Monitoring System** is an AI-powered quality control platform that monitors a multi-station manufacturing production line in real time. It combines:

- **YOLOv8 object detection** to identify manufacturing defects and worker PPE compliance violations
- **LLM-based root-cause analysis** to automatically diagnose the cause of each detected anomaly and recommend corrective actions
- **Multi-station simulation** of realistic factory telemetry (vibration, temperature, pressure, conveyor speed)

Built for the **Cognizant Technoverse Hackathon**.

---

## 🏗️ Architecture

```
                   ┌─────────────────────────────────────────┐
                   │        Production Line Simulator         │
                   │  Station 1   Station 2   Station 3   Station 4  │
                   │  CNC Mill  Thermal Tmt  Robotic Asm  QA Zone  │
                   └───────────────────┬─────────────────────┘
                                       │ InspectionTargets + Telemetry
                                       ▼
                   ┌─────────────────────────────────────────┐
                   │        YOLOv8 Detection Engine          │
                   │  • Defect Detection (surface scratch,   │
                   │    hairline crack, dimensional warp,    │
                   │    weld porosity, misalignment)         │
                   │  • PPE Compliance (helmet, hi-vis vest) │
                   └───────────────────┬─────────────────────┘
                                       │ Detection Objects
                                       ▼
                   ┌─────────────────────────────────────────┐
                   │       LLM Root-Cause Analyzer           │
                   │  • Ollama (local LLM) if available      │
                   │  • Template engine fallback (no deps)   │
                   │  • Outputs: cause, action, severity,    │
                   │    confidence score                     │
                   └───────────────────┬─────────────────────┘
                                       │ RootCauseReports
                                       ▼
                   ┌─────────────────────────────────────────┐
                   │         Alert Reporter                  │
                   │  • Rich ANSI console output             │
                   │  • JSON reports → outputs/              │
                   └─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Akhilesh-0529/simulating-and-detecting-anomaly-in-manufacturing-production-chain-.git
cd simulating-and-detecting-anomaly-in-manufacturing-production-chain-

# Run (stdlib only — no installation required for simulation mode)
python main.py
```

### Optional: Install YOLOv8 for Real Inference

```bash
pip install -r requirements.txt
python main.py
```

---

## 🔍 Detection Pipeline

### Manufacturing Defect Classes
| Class | Description |
|---|---|
| `surface_scratch` | Tooling wear, chip re-cutting |
| `hairline_crack` | Thermal stress, hydrogen embrittlement |
| `dimensional_warp` | Furnace non-uniformity, cooling asymmetry |
| `weld_porosity` | Shielding gas contamination, surface oxidation |
| `component_misalignment` | Robot TCP drift, fixture wear |

### PPE Compliance Classes
| Class | Severity |
|---|---|
| `missing_helmet` | CRITICAL |
| `missing_vest` | HIGH |

---

## 🤖 LLM Root-Cause Analysis

Each detected anomaly is passed to the LLM analyzer, which outputs:
- **Root cause** — domain-specific engineering explanation
- **Recommended corrective action** — immediate and preventive steps
- **Severity** — `LOW / MEDIUM / HIGH / CRITICAL`
- **Confidence score** — 0.0–1.0

The analyzer tries a **local Ollama LLM** (`llama3.2`) first, then falls back to a rich **template engine** that simulates expert-level analysis — no internet connection needed.

---

## 📡 Simulated Production Scenarios

| Cycle | Scenario | Injected Anomaly |
|---|---|---|
| 01 | Normal operations | None |
| 02 | CNC spindle wear | `spindle_wear_vibration` → Station 1 |
| 03 | Dual failure | `thermal_furnace_drift` + `worker_ppe_violation` |
| 04 | Pneumatic drop | `pneumatic_pressure_drop` → Station 3 |
| 05 | Recovery | None |

---

## 📁 Project Structure

```
.
├── main.py                        # Main entry point
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py                # Station configs, defect/PPE classes
├── simulation/
│   ├── __init__.py
│   └── production_chain.py        # Multi-station simulator + telemetry
├── detection/
│   ├── yolo_detector.py           # YOLOv8 detector (with simulation fallback)
│   └── detector.py                # Original anomaly detection module
├── analysis/
│   ├── __init__.py
│   └── root_cause_llm.py          # LLM root-cause analyzer (Ollama + templates)
├── reporting/
│   ├── __init__.py
│   └── alert_reporter.py          # ANSI console + JSON reporter
├── outputs/                       # JSON reports saved here per cycle
├── data/                          # Original time-series data pipeline
├── features/
├── evaluation/
└── run_demo.py                    # Original FOD demo (preserved)
```

---

## 🛠️ Tech Stack

- **YOLOv8** (Ultralytics) — Object detection backbone
- **OpenCV** — Frame preprocessing and visualization
- **LLMs** — Ollama (`llama3.2`) for root-cause analysis
- **Python 3.8+** — Stdlib-only core (zero required installs for simulation mode)

---

## 📊 Sample Output

```
══════════════════════════════════════════════════════════════════════
  SMART FACTORY MONITORING SYSTEM
  YOLOv8 Object Detection + LLM Root-Cause Analysis
  Cognizant Technoverse Hackathon  |  2026-11-18 01:03:07
══════════════════════════════════════════════════════════════════════

  CYCLE 02 | Batch: CYCLE_02_VIBRATION ⚠️

  ◆ STATION_1 — CNC Milling
    Telemetry │ Temperature C: 27.30 │ Vibration Mm S: 6.60 │ Status: WARNING
    Detections (3):
      • surface_scratch       conf=0.87  bbox=[312, 198, 362, 223]
      • hairline_crack        conf=0.74  bbox=[198, 312, 248, 337]
      • manufactured_part     conf=0.95  bbox=[120, 140, 520, 480]

    [LLM Analysis — template_engine]  🟠 HIGH
    Confidence: 91.2%  |  0 ms
    Root Cause:
      Tool wear on CNC spindle exceeded threshold — cutting insert
      degraded beyond 0.3 mm flank wear.
    Action:
      Replace cutting insert immediately. Inspect spindle bearing
      pre-load. Flush coolant nozzles.
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Built with ❤️ for the Cognizant Technoverse Hackathon*
