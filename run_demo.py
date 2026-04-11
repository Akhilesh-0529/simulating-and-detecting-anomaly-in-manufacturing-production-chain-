"""End-to-end demo. Run with: python run_demo.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from data.generator import generate_session
from features.extractor import sliding_window_features, label_windows
from detection.detector import ForeignObjectDetector
from evaluation.metrics import evaluate_detection, print_report

def main():
    rng = np.random.default_rng(0)
    print("\n1. Generating synthetic sensor data ...")
    signals, events = generate_session(duration_s=10.0, n_events=5, rng=rng)
    print(f"   {len(signals):,} samples  |  {len(events)} injected events\n")

    print("2. Sliding-window feature extraction ...")
    X, centers = sliding_window_features(signals, window_ms=50, step_ms=5)
    y = label_windows(centers, events, lookahead_ms=20)
    print(f"   {X.shape[0]} windows x {X.shape[1]} features  |  {y.sum()} event windows\n")

    print("3. Fitting semi-supervised detector ...")
    det = ForeignObjectDetector(n_estimators=200, threshold_quantile=0.97)
    det.fit(X, y)
    y_pred = det.predict(X)
    print(f"   Anomaly threshold: {det.threshold_:.4f}\n")

    print("4. Evaluating ...")
    result = evaluate_detection(y_pred, y, centers, events, window_ms=50)
    print_report(result)

if __name__ == "__main__":
    main()
