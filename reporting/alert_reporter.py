"""
Factory Alert Reporter
=======================
Formats and outputs detection + LLM analysis results to console and JSON.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


# ANSI color codes (stdlib only)
class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"


_SEVERITY_COLOR = {
    "CRITICAL": _C.RED + _C.BOLD,
    "HIGH":     _C.RED,
    "MEDIUM":   _C.YELLOW,
    "LOW":      _C.GREEN,
}

_SEVERITY_ICON = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}


def _sev_color(severity: str, text: str) -> str:
    color = _SEVERITY_COLOR.get(severity, _C.WHITE)
    return f"{color}{text}{_C.RESET}"


class FactoryAlertReporter:
    """
    Rich console + JSON reporter for the Smart Factory Monitoring System.

    Usage:
        reporter = FactoryAlertReporter(output_dir="outputs")
        reporter.report_cycle(batch_id, targets, detections_map, telemetry_map, root_causes_map)
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._total_defects = 0
        self._total_ppe = 0
        self._critical_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_header(self, title: str = "SMART FACTORY MONITORING SYSTEM") -> None:
        width = 70
        print(f"\n{_C.CYAN}{_C.BOLD}{'═' * width}{_C.RESET}")
        print(f"{_C.CYAN}{_C.BOLD}  {title}{_C.RESET}")
        print(f"{_C.CYAN}  YOLOv8 Object Detection + LLM Root-Cause Analysis{_C.RESET}")
        print(f"{_C.DIM}  Cognizant Technoverse Hackathon  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{_C.RESET}")
        print(f"{_C.CYAN}{'═' * width}{_C.RESET}\n")

    def report_cycle(
        self,
        cycle_id: int,
        batch_id: str,
        stations: List[str],
        detections_map: Dict[str, List[Dict[str, Any]]],
        telemetry_map: Dict[str, Dict[str, Any]],
        root_causes_map: Dict[str, Any],
    ) -> None:
        """Print a full cycle report and save JSON."""
        ts = datetime.now().strftime("%H:%M:%S")
        all_clear = all(len(v) == 0 for v in detections_map.values())

        print(f"{_C.BOLD}{'─' * 70}{_C.RESET}")
        status_icon = "✅" if all_clear else "⚠️ "
        print(f"{_C.BOLD}  CYCLE {cycle_id:02d}  |  Batch: {batch_id}  |  {ts}  {status_icon}{_C.RESET}")
        print(f"{'─' * 70}")

        cycle_data: Dict[str, Any] = {
            "cycle_id": cycle_id,
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "stations": [],
        }

        for station in stations:
            detections = detections_map.get(station, [])
            telemetry = telemetry_map.get(station, {})
            rca_reports = root_causes_map.get(station, [])

            self._print_station_block(station, detections, telemetry, rca_reports)

            station_data = {
                "station": station,
                "telemetry": telemetry,
                "detections": detections,
                "root_cause_analyses": [
                    {
                        "defect_type": r.defect_type,
                        "severity": r.severity,
                        "root_cause": r.root_cause,
                        "recommended_action": r.recommended_action,
                        "confidence": r.confidence,
                        "provider": r.llm_provider,
                    }
                    for r in rca_reports
                ],
            }
            cycle_data["stations"].append(station_data)

            # Accumulate totals
            for d in detections:
                cls = d.get("class", "")
                if "missing_" in cls or "no_" in cls:
                    self._total_ppe += 1
                else:
                    self._total_defects += 1

        if all_clear:
            print(f"\n  {_C.GREEN}✓ All stations nominal — no anomalies detected.{_C.RESET}\n")

        self.save_json_report(batch_id, cycle_id, cycle_data)

    def print_summary(self, total_cycles: int) -> None:
        """Print end-of-run statistics."""
        width = 70
        print(f"\n{_C.CYAN}{_C.BOLD}{'═' * width}{_C.RESET}")
        print(f"{_C.CYAN}{_C.BOLD}  RUN SUMMARY{_C.RESET}")
        print(f"{'─' * width}")
        print(f"  Cycles Executed       : {total_cycles}")
        print(f"  Defects Detected      : {_C.RED if self._total_defects else _C.GREEN}{self._total_defects}{_C.RESET}")
        print(f"  PPE Violations        : {_C.RED if self._total_ppe else _C.GREEN}{self._total_ppe}{_C.RESET}")
        print(f"  Reports saved to      : {_C.DIM}{self.output_dir}/{_C.RESET}")
        print(f"{_C.CYAN}{'═' * width}{_C.RESET}\n")

    def save_json_report(self, batch_id: str, cycle_id: int, data: Dict[str, Any]) -> None:
        """Persist cycle report as JSON."""
        filename = os.path.join(self.output_dir, f"report_{batch_id}_cycle{cycle_id:02d}.json")
        with open(filename, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_station_block(
        self,
        station: str,
        detections: List[Dict[str, Any]],
        telemetry: Dict[str, Any],
        rca_reports: List[Any],
    ) -> None:
        has_issues = len(detections) > 0
        prefix = f"{_C.RED}  ◆{_C.RESET}" if has_issues else f"{_C.GREEN}  ◇{_C.RESET}"
        print(f"\n{prefix} {_C.BOLD}{station}{_C.RESET}")

        # Telemetry row
        tel_parts = []
        for k, v in telemetry.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                tel_parts.append(f"{label}: {v:.2f}")
            else:
                tel_parts.append(f"{label}: {v}")
        if tel_parts:
            print(f"    {_C.DIM}Telemetry │ {' │ '.join(tel_parts)}{_C.RESET}")

        if not detections:
            print(f"    {_C.GREEN}No anomalies detected.{_C.RESET}")
            return

        # Detections table
        print(f"    {_C.YELLOW}Detections ({len(detections)}):{_C.RESET}")
        for det in detections:
            cls    = det.get("class", "unknown")
            conf   = det.get("confidence", 0.0)
            bbox   = det.get("bbox", [])
            bbox_s = f"[{', '.join(str(int(b)) for b in bbox)}]" if bbox else "N/A"
            line   = f"      • {_C.BOLD}{cls}{_C.RESET}  conf={conf:.2f}  bbox={bbox_s}"
            print(line)

        # LLM RCA blocks
        for report in rca_reports:
            sev_str = _sev_color(report.severity, f"{_SEVERITY_ICON.get(report.severity, '')} {report.severity}")
            print(f"\n    {_C.MAGENTA}[LLM Analysis — {report.llm_provider}]{_C.RESET}  {sev_str}")
            print(f"    {_C.DIM}Confidence: {report.confidence * 100:.1f}%  |  {report.analysis_time_ms:.0f} ms{_C.RESET}")
            print(f"    {_C.CYAN}Root Cause:{_C.RESET}")
            for line in self._wrap(report.root_cause, 65):
                print(f"      {line}")
            print(f"    {_C.CYAN}Action:{_C.RESET}")
            for line in self._wrap(report.recommended_action, 65):
                print(f"      {line}")

    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """Simple word-wrap without textwrap dependency."""
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]
