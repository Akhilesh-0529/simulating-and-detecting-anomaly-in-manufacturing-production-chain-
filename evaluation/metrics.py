"""
Event-level detection metrics: TPR, FPR, advance detection time.

These are *event-level* metrics (not sample-level), which is the
correct evaluation unit for foreign object detection use cases.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    tpr: float
    fpr: float
    advance_detection_ms: list = field(default_factory=list)
    n_events: int = 0
    n_detected: int = 0
    n_false_alarms: int = 0
    mean_advance_ms: float = 0.0
    median_advance_ms: float = 0.0


def evaluate_detection(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    centers_ms: np.ndarray,
    events: pd.DataFrame,
    window_ms: int = 50,
) -> DetectionResult:
    """
    Event-level evaluation of detector output.

    Parameters
    ----------
    y_pred      : binary predictions per window
    y_true      : binary ground truth per window
    centers_ms  : center sample index of each window
    events      : DataFrame with columns [start_ms, end_ms]
    window_ms   : window size used during feature extraction
    """
    n_events = len(events)
    detected, advance_times = [], []

    for _, ev in events.iterrows():
        ev_peak = (ev["start_ms"] + ev["end_ms"]) // 2
        pre_mask = (centers_ms <= ev_peak) & (centers_ms >= ev["start_ms"] - window_ms)
        pre_preds = y_pred[pre_mask]
        pre_centers = centers_ms[pre_mask]

        if np.any(pre_preds == 1):
            first_det = pre_centers[pre_preds == 1][0]
            advance_times.append(float(ev_peak - first_det))
            detected.append(True)
        else:
            detected.append(False)

    n_detected = sum(detected)
    normal_mask = y_true == 0
    n_fa = int(np.sum(y_pred[normal_mask] == 1))

    return DetectionResult(
        tpr=n_detected / max(n_events, 1),
        fpr=n_fa / max(int(np.sum(normal_mask)), 1),
        advance_detection_ms=advance_times,
        n_events=n_events,
        n_detected=n_detected,
        n_false_alarms=n_fa,
        mean_advance_ms=float(np.mean(advance_times)) if advance_times else 0.0,
        median_advance_ms=float(np.median(advance_times)) if advance_times else 0.0,
    )


def print_report(result: DetectionResult):
    print("=" * 44)
    print("  Detection Performance Report")
    print("=" * 44)
    print(f"  Events in session       : {result.n_events}")
    print(f"  Detected (TP)           : {result.n_detected}")
    print(f"  True positive rate      : {result.tpr:.1%}")
    print(f"  False positive rate     : {result.fpr:.2%}")
    print(f"  False alarms            : {result.n_false_alarms}")
    print(f"  Mean advance time (ms)  : {result.mean_advance_ms:.1f}")
    print(f"  Median advance time(ms) : {result.median_advance_ms:.1f}")
    print("=" * 44)
