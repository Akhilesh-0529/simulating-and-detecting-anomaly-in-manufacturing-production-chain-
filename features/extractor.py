"""
Sliding-window feature extraction for multivariate sensor streams.
Compatible with sktime's time series classification / annotation APIs.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

FEATURE_NAMES = [
    "rms", "peak", "crest_factor", "kurtosis", "skewness",
    "zcr", "energy_ratio_high", "var"
]


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract a fixed feature vector from a single (n_timesteps, n_sensors) window.

    Features per channel: RMS, peak, crest factor, kurtosis, skewness,
    zero-crossing rate, high-frequency energy ratio, variance.
    All channels are concatenated into a flat vector.
    """
    n_features = len(FEATURE_NAMES)
    n_channels = window.shape[1] if window.ndim > 1 else 1
    feats = np.zeros(n_channels * n_features)

    for ch in range(n_channels):
        x = window[:, ch] if window.ndim > 1 else window
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        crest = peak / (rms + 1e-9)
        kurt = float(kurtosis(x))
        sk = float(skew(x))
        zcr = float(np.mean(np.diff(np.sign(x)) != 0))
        fft = np.abs(np.fft.rfft(x))
        mid = len(fft) // 2
        energy_ratio = float(np.sum(fft[mid:] ** 2) / (np.sum(fft ** 2) + 1e-9))
        var = float(np.var(x))
        feats[ch * n_features:(ch + 1) * n_features] = [
            rms, peak, crest, kurt, sk, zcr, energy_ratio, var
        ]
    return feats


def sliding_window_features(
    signals: pd.DataFrame,
    window_ms: int = 50,
    step_ms: int = 10,
):
    """
    Apply sliding-window feature extraction over the full signal dataframe.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_features)
    centers_ms : np.ndarray — center sample index of each window
    """
    arr = signals.values
    n = len(arr)
    rows, centers = [], []
    for start in range(0, n - window_ms, step_ms):
        end = start + window_ms
        rows.append(extract_window_features(arr[start:end]))
        centers.append((start + end) // 2)
    return np.array(rows), np.array(centers)


def label_windows(
    centers_ms: np.ndarray,
    events: pd.DataFrame,
    lookahead_ms: int = 0,
) -> np.ndarray:
    """
    Binary-label each window: 1 if the window center falls within an event
    (optionally shifted by `lookahead_ms` for early-detection training).
    """
    y = np.zeros(len(centers_ms), dtype=int)
    for _, ev in events.iterrows():
        start = ev["start_ms"] - lookahead_ms
        end = ev["end_ms"]
        y[(centers_ms >= start) & (centers_ms <= end)] = 1
    return y
