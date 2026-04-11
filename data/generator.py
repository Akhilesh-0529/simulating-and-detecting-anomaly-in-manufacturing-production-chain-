"""
Synthetic multivariate sensor data generator.
Simulates vibration, acoustic and mechanical sensor streams
at millisecond granularity, with injected foreign object events.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EventConfig:
    duration_ms: int = 120
    rise_ms: int = 40
    amplitude_vib: float = 1.8
    amplitude_aco: float = 2.1
    amplitude_mec: float = 1.3


@dataclass
class SensorConfig:
    sample_rate_hz: int = 1000
    noise_std_vib: float = 0.08
    noise_std_aco: float = 0.06
    noise_std_mec: float = 0.05
    baseline_vib: float = 0.15
    baseline_aco: float = 0.10
    baseline_mec: float = 0.20


def generate_session(duration_s=5.0, n_events=3, sensor_cfg=None, event_cfg=None, rng=None):
    """
    Generate one recording session with `n_events` injected foreign object events.

    Returns
    -------
    signals : pd.DataFrame  shape (n_samples, 3) — [vibration, acoustic, mechanics]
    events  : pd.DataFrame  columns [start_ms, end_ms] — ground-truth windows
    """
    if rng is None:
        rng = np.random.default_rng(42)
    sc = sensor_cfg or SensorConfig()
    ec = event_cfg or EventConfig()
    n = int(duration_s * sc.sample_rate_hz)
    t_ms = np.arange(n)

    # Baseline: band-limited noise around machine operating point
    vib = sc.baseline_vib + rng.normal(0, sc.noise_std_vib, n)
    aco = sc.baseline_aco + rng.normal(0, sc.noise_std_aco, n)
    mec = sc.baseline_mec + rng.normal(0, sc.noise_std_mec, n)

    # Low-frequency machine vibration (realistic background)
    vib += 0.05 * np.sin(2 * np.pi * 12 * t_ms / sc.sample_rate_hz)
    mec += 0.03 * np.sin(2 * np.pi * 8 * t_ms / sc.sample_rate_hz + 0.7)

    # Inject events non-overlapping in the middle 80% of the session
    event_records = []
    event_dur = ec.duration_ms
    min_gap = 500
    margin = int(0.1 * n)
    candidates = np.sort(
        rng.choice(range(margin, n - margin - event_dur), size=n_events * 5, replace=False)
    )
    placed = []
    for c in candidates:
        if all(abs(c - p) > event_dur + min_gap for p in placed):
            placed.append(c)
        if len(placed) == n_events:
            break

    for start in placed:
        end = start + event_dur
        prog = np.linspace(0, 1, event_dur)
        burst = np.sin(prog * np.pi)
        ramp = np.where(prog < ec.rise_ms / event_dur, prog / (ec.rise_ms / event_dur), 1.0)
        envelope = burst * ramp

        vib[start:end] += ec.amplitude_vib * envelope * (1 + rng.normal(0, 0.1, event_dur))
        aco[start:end] += ec.amplitude_aco * envelope * (1 + rng.normal(0, 0.08, event_dur))
        mec[start:end] += ec.amplitude_mec * envelope * (1 + rng.normal(0, 0.12, event_dur))
        event_records.append({"start_ms": start, "end_ms": end})

    idx = pd.date_range("2024-01-01", periods=n, freq="ms")
    signals = pd.DataFrame({"vibration": vib, "acoustic": aco, "mechanics": mec}, index=idx)
    events = pd.DataFrame(event_records)
    return signals, events
