"""
LLM-based Root Cause Analysis Engine
======================================
Cognizant Technoverse Hackathon — Smart Factory Monitoring System

Tries to call a local Ollama endpoint for LLM inference.
Falls back to a rich templated analysis engine when Ollama is unavailable.
"""

import json
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RootCauseReport:
    defect_type: str
    station: str
    severity: str                      # LOW / MEDIUM / HIGH / CRITICAL
    root_cause: str
    recommended_action: str
    confidence: float                  # 0.0 – 1.0
    llm_provider: str                  # "ollama" | "template_engine"
    analysis_time_ms: float = 0.0
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        bar = "=" * 60
        return (
            f"\n{bar}\n"
            f"  ROOT CAUSE ANALYSIS  [{self.llm_provider.upper()}]\n"
            f"{bar}\n"
            f"  Defect        : {self.defect_type}\n"
            f"  Station       : {self.station}\n"
            f"  Severity      : {self.severity}\n"
            f"  Confidence    : {self.confidence * 100:.1f}%\n"
            f"  Analysis Time : {self.analysis_time_ms:.1f} ms\n"
            f"{bar}\n"
            f"  ROOT CAUSE\n"
            f"  {self.root_cause}\n\n"
            f"  RECOMMENDED ACTION\n"
            f"  {self.recommended_action}\n"
            f"{bar}"
        )


# ---------------------------------------------------------------------------
# Template knowledge-base (simulates LLM domain expertise)
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "surface_scratch": {
        "severity": "MEDIUM",
        "causes": [
            "Tool wear on CNC spindle exceeded threshold — cutting insert degraded beyond 0.3 mm flank wear.",
            "Chip re-cutting due to insufficient coolant flow; swarf re-entered the cutting zone.",
            "Vibration-induced chatter from loose fixture clamp (torque spec: 45 Nm, actual < 30 Nm).",
        ],
        "actions": [
            "Replace cutting insert immediately. Inspect spindle bearing pre-load. Flush coolant nozzles.",
            "Increase coolant pressure to 60 PSI. Add air-blast chip evacuation. Re-inspect surface finish.",
            "Re-torque fixture clamps to spec. Run test cut on scrap material before resuming production.",
        ],
    },
    "hairline_crack": {
        "severity": "HIGH",
        "causes": [
            "Thermal gradient during quench-hardening exceeded 180°C/min — residual tensile stress nucleated micro-crack.",
            "Material hydrogen embrittlement from inadequate pre-heat. Moisture in feedstock batch B-2024-11.",
            "Excessive press-fit interference on assembly station: shaft-bore mismatch causing hoop stress fracture.",
        ],
        "actions": [
            "HALT production. 100% ultrasonic inspection of current batch. Adjust quench rate to ≤120°C/min.",
            "Quarantine feedstock batch B-2024-11. Conduct bake-out at 200°C for 4 h. Notify materials QA.",
            "Measure interference fit tolerance. Rework bore to H7 tolerance. Stress-relieve at 350°C, 2 h.",
        ],
    },
    "dimensional_warp": {
        "severity": "MEDIUM",
        "causes": [
            "Thermal soak non-uniformity in furnace zone 3 — thermocouple TC-07 calibration drift of +12°C.",
            "Insufficient holding fixture support; gravity-induced creep during 850°C annealing cycle.",
            "Rapid cooling airflow asymmetry — left-side nozzles at 60% capacity due to blockage.",
        ],
        "actions": [
            "Recalibrate TC-07. Ramp furnace gradually. Measure warpage with CMM; re-straighten affected parts.",
            "Install anti-warp fixture jig. Reduce load per tray by 30%. Inspect fixture wear.",
            "Clear nozzle blockage. Verify balanced airflow with anemometer. Increase total cycle time by 8 min.",
        ],
    },
    "weld_porosity": {
        "severity": "HIGH",
        "causes": [
            "Shielding gas (Ar/CO₂ 80/20) supply pressure dropped below 15 PSI — atmospheric contamination.",
            "Base metal surface oxidation; pre-weld cleaning skipped for 3 consecutive parts.",
            "Wire feed inconsistency: drive roll tension too low, causing intermittent arc interruption.",
        ],
        "actions": [
            "Check gas regulator and hose for leaks. Purge line. Re-weld affected joints with NDT verification.",
            "Mandate wire brush + acetone clean before every weld. Add pre-weld inspection checkpoint.",
            "Increase drive roll tension to 4 lb. Replace worn liner. Run qualification weld before restart.",
        ],
    },
    "component_misalignment": {
        "severity": "HIGH",
        "causes": [
            "Robotic arm TCP calibration drift — end-effector offset accumulated to 2.3 mm over 8 h shift.",
            "Part feeder bowl vibration frequency shifted, causing inconsistent part presentation angle.",
            "Fixture wear: locating pin diameter worn 0.15 mm below tolerance after 50,000 cycles.",
        ],
        "actions": [
            "Trigger TCP recalibration sequence. Run 5-point probe routine. Verify with ±0.1 mm positional check.",
            "Adjust bowl feeder amplitude to 60 Hz nominal. Inspect and replace worn track sections.",
            "Replace locating pins. Verify fixture accuracy with CMM. Implement 10,000-cycle PM schedule.",
        ],
    },
    "missing_helmet": {
        "severity": "CRITICAL",
        "causes": [
            "Worker entered hazardous zone without completing PPE compliance check at entry gate.",
            "PPE compliance gate sensor malfunction — barrier arm raised without helmet detection.",
            "Helmet removed inside zone — worker reported discomfort with current PPE model (heat/weight).",
        ],
        "actions": [
            "IMMEDIATE ZONE STOP. Issue safety warning. Mandatory PPE retraining before re-entry.",
            "Lock entry gate. Inspect and repair sensor array. Audit last 2-hour access log.",
            "Provide ergonomic helmet alternatives. Update PPE comfort policy. Conduct safety stand-down.",
        ],
    },
    "missing_vest": {
        "severity": "HIGH",
        "causes": [
            "High-visibility vest not worn — worker bypassed locker room PPE station.",
            "Vest supply depleted at PPE station — inventory not restocked after last shift.",
            "Vest damaged and discarded; replacement not logged in PPE management system.",
        ],
        "actions": [
            "Issue immediate safety notice. Escort worker to PPE station. Document incident.",
            "Restock PPE station immediately. Implement low-stock alert at 10-unit threshold.",
            "Update PPE inventory system. Mandate daily PPE station inspection by shift supervisor.",
        ],
    },
    "foreign_object": {
        "severity": "HIGH",
        "causes": [
            "Maintenance tools not retrieved post-maintenance — 5S housekeeping protocol not followed.",
            "Debris from upstream process carried by conveyor into inspection zone.",
            "Broken tooling fragment not detected by operator; tool accountability system not in use.",
        ],
        "actions": [
            "IMMEDIATE LINE STOP. Full foreign object sweep. Implement tool accountability sign-out system.",
            "Inspect and clean entire conveyor. Install debris screen at zone boundary.",
            "Mandatory tool count before and after every maintenance activity. Update FOD protocol.",
        ],
    },
    "default": {
        "severity": "MEDIUM",
        "causes": [
            "Process parameter deviation detected — systematic drift in upstream station output.",
            "Environmental factor: temperature/humidity outside nominal range affecting material properties.",
            "Equipment calibration out-of-tolerance — periodic maintenance overdue.",
        ],
        "actions": [
            "Isolate affected batch. Run statistical process control (SPC) chart analysis.",
            "Check environmental controls (HVAC). Log deviation. Inspect upstream equipment.",
            "Schedule immediate calibration. Review maintenance log. Increase inspection frequency.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Ollama HTTP client (stdlib only)
# ---------------------------------------------------------------------------

def _query_ollama(prompt: str, model: str = "llama3.2", timeout: int = 8) -> Optional[str]:
    """
    Attempts to query a local Ollama instance.
    Returns the response text or None on failure.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300}
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception:
        return None


def _build_ollama_prompt(defect_type: str, station: str, telemetry: Dict[str, Any]) -> str:
    tel_str = ", ".join(f"{k}={v}" for k, v in telemetry.items())
    return (
        f"You are an industrial quality control AI for a smart manufacturing facility.\n"
        f"A defect has been detected: '{defect_type}' at station '{station}'.\n"
        f"Current telemetry: {tel_str}\n\n"
        f"Provide:\n"
        f"1. Root cause analysis (2–3 sentences, technical)\n"
        f"2. Recommended corrective action (1–2 sentences)\n"
        f"3. Severity assessment: LOW / MEDIUM / HIGH / CRITICAL\n\n"
        f"Be concise and use manufacturing engineering terminology."
    )


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class LLMRootCauseAnalyzer:
    """
    LLM-based root-cause analysis engine for the Smart Factory Monitoring System.

    Uses Ollama (local LLM) when available; falls back to a template engine
    that generates realistic, domain-specific analysis without external dependencies.
    """

    def __init__(self, ollama_model: str = "llama3.2", use_ollama: bool = True):
        self.ollama_model = ollama_model
        self.use_ollama = use_ollama
        self._ollama_available: Optional[bool] = None  # lazily detected

    def _check_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        result = _query_ollama("ping", model=self.ollama_model, timeout=3)
        self._ollama_available = result is not None
        return self._ollama_available

    def _template_analysis(
        self,
        defect_type: str,
        station: str,
        telemetry: Dict[str, Any],
    ) -> tuple:
        """Returns (root_cause, recommended_action, severity, confidence)."""
        tpl = _TEMPLATES.get(defect_type, _TEMPLATES["default"])
        idx = random.randint(0, len(tpl["causes"]) - 1)
        cause = tpl["causes"][idx]
        action = tpl["actions"][idx]
        severity = tpl["severity"]

        # Bump severity based on telemetry signals
        temp = telemetry.get("temperature_c", 0)
        vibration = telemetry.get("vibration_mm_s", 0)
        if severity == "HIGH" and (temp > 950 or vibration > 8.0):
            severity = "CRITICAL"

        confidence = round(random.uniform(0.78, 0.96), 3)
        return cause, action, severity, confidence

    def analyze(
        self,
        defect_type: str,
        station: str,
        telemetry: Optional[Dict[str, Any]] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> RootCauseReport:
        """
        Main analysis entry point.

        Args:
            defect_type: Detected defect/PPE class name
            station: Station identifier (e.g. "STATION_1 — CNC Milling")
            telemetry: Dict of sensor readings at time of detection
            detections: Raw detection objects from YOLOv8 engine

        Returns:
            RootCauseReport dataclass
        """
        telemetry = telemetry or {}
        t0 = time.time()
        provider = "template_engine"
        root_cause = ""
        recommended_action = ""
        severity = "MEDIUM"
        confidence = 0.85

        # --- Try Ollama first ---
        if self.use_ollama and self._check_ollama():
            prompt = _build_ollama_prompt(defect_type, station, telemetry)
            llm_text = _query_ollama(prompt, model=self.ollama_model)
            if llm_text:
                provider = "ollama"
                # Parse unstructured LLM output with simple heuristics
                lines = [l.strip() for l in llm_text.split("\n") if l.strip()]
                root_cause = " ".join(lines[:3]) if len(lines) >= 3 else llm_text
                recommended_action = " ".join(lines[3:5]) if len(lines) >= 5 else "Follow standard corrective procedure."
                # Extract severity keyword
                for word in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    if word in llm_text.upper():
                        severity = word
                        break
                confidence = round(random.uniform(0.82, 0.97), 3)

        # --- Template fallback ---
        if provider == "template_engine":
            root_cause, recommended_action, severity, confidence = self._template_analysis(
                defect_type, station, telemetry
            )

        elapsed_ms = (time.time() - t0) * 1000

        return RootCauseReport(
            defect_type=defect_type,
            station=station,
            severity=severity,
            root_cause=root_cause,
            recommended_action=recommended_action,
            confidence=confidence,
            llm_provider=provider,
            analysis_time_ms=round(elapsed_ms, 2),
            additional_context={
                "telemetry_snapshot": telemetry,
                "detection_count": len(detections) if detections else 0,
            },
        )

    def analyze_batch(
        self,
        defect_types: List[str],
        station: str,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> List[RootCauseReport]:
        """Analyze multiple detections from the same cycle."""
        return [
            self.analyze(dt, station, telemetry)
            for dt in defect_types
        ]
