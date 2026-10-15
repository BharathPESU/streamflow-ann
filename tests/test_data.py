import numpy as np, pandas as pd
from src.data import build_windows, prepare, make_synthetic
from src.metrics import evaluate


def test_window_alignment():
    rain = np.arange(100, 120, dtype=float)       # R(t) = 100 + t
    runoff = np.arange(0, 20, dtype=float)        # Q(t) = t
    X, y, ok, start = build_windows(rain, runoff, lag_rain=2, lag_runoff=3)
    assert start == 3
    # first target is Q(3); features R(2),R(1),Q(2),Q(1),Q(0)
    assert y[0] == 3
    assert list(X[0]) == [102, 101, 2, 1, 0]


def test_no_future_leakage_and_split():
    ds = prepare(make_synthetic(), 3, 4)
    assert ds.X_train.shape[1] == 7
    assert len(ds.X_val) > 1500 and len(ds.X_train) > 1500
    assert ds.X_train.max() <= 1.0 + 1e-9       # scaled with train max


def test_metrics_perfect_prediction():
    y = np.array([0.01, 0.2, 0.5])
    m = evaluate(y, y, q_max=10000)
    assert m["overall"] == 0
