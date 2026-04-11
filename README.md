# Foreign Object Detection —  Demo Portfolio Project

A multivariate time-series detection pipeline built with `sktime` and `scikit-learn`,
demonstrating early event detection on synthetic sensor data (vibration, acoustics, mechanics).

Mirrors the goals of the European Summer of Code agricultural machinery project:
- Multivariate sensor streams at millisecond granularity
- Feature extraction pipeline (RMS, kurtosis, crest factor, ZCR, FFT energy ratio)
- Semi-supervised Isolation Forest detector (trains on normal data only)
- Transparent metrics: TPR, FPR, advance detection time

## Structure

```
fod/
├── data/generator.py          synthetic sensor data with injected events
├── features/extractor.py      sliding-window feature extraction
├── detection/detector.py      sklearn-compatible semi-supervised detector
├── evaluation/metrics.py      event-level TPR, FPR, advance-detection-time
├── run_demo.py                end-to-end pipeline demo
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

## Extending with sktime

The feature matrix produced by `sliding_window_features()` is directly compatible
with sktime's `TimeSeriesForestClassifier`, `ROCKET`, or `WEASEL` classifiers.
Swap in any sktime classifier at the `ForeignObjectDetector` fit step to benchmark
algorithmic approaches.

```python
from sktime.classification.interval_based import TimeSeriesForestClassifier
# wrap signals into sktime panel format and go!
```
