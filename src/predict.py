"""CLI inference tool to test trained Feed-Forward and LSTM streamflow models.

Examples:
    # 1. Custom What-If scenario (past 3 days rain, past 4 days runoff)
    python predict.py --rain 0.5 1.8 0.2 --runoff 300 450 1100 1650

    # 2. Test against a real historical date from the dataset
    python predict.py --date 2008-03-20
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch

from src.models import FeedForward, LSTMNet
from src.data import load_csv, clean


Q_MAX = 29700.0
R_MAX = 6.96
PEAK_THRESHOLD = 1500.0


def load_models(results_dir="results"):
    ff_path = os.path.join(results_dir, "model_feedforward.pt")
    lstm_path = os.path.join(results_dir, "model_lstm.pt")
    models = {}
    if os.path.exists(ff_path):
        m = FeedForward(n_in=7, hidden=12)
        m.load_state_dict(torch.load(ff_path, weights_only=True))
        m.eval()
        models["feedforward"] = m
    if os.path.exists(lstm_path):
        m = LSTMNet(hidden=12)
        m.load_state_dict(torch.load(lstm_path, weights_only=True))
        m.eval()
        models["lstm"] = m
    return models


def predict_ff(model, rain_3, runoff_4):
    """Features: [R(t-1), R(t-2), R(t-3), Q(t-1), Q(t-2), Q(t-3), Q(t-4)] scaled."""
    features = np.concatenate([
        np.array(rain_3, dtype=float) / R_MAX,
        np.array(runoff_4, dtype=float) / Q_MAX
    ])
    inp = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(inp).item()
    return max(0.0, float(out * Q_MAX))


def predict_lstm(model, rain_5, runoff_5):
    """Features: sequence of 5 steps, each [R, Q] (oldest first: t-5 .. t-1)."""
    r = np.array(rain_5[::-1], dtype=float) / R_MAX   # oldest first
    q = np.array(runoff_5[::-1], dtype=float) / Q_MAX
    seq = np.stack([r, q], axis=-1)
    inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(inp).item()
    return max(0.0, float(out * Q_MAX))


def main():
    ap = argparse.ArgumentParser(description="Test trained streamflow forecasting models.")
    ap.add_argument("--date", help="Test on a specific date (YYYY-MM-DD) from data/leaf_river.csv")
    ap.add_argument("--data", default="data/leaf_river.csv", help="Path to leaf_river.csv")
    ap.add_argument("--rain", nargs="+", type=float,
                    help="Recent rainfall in inches: R(t-1), R(t-2), ...")
    ap.add_argument("--runoff", nargs="+", type=float,
                    help="Recent runoff in ft^3/s: Q(t-1), Q(t-2), ...")
    ap.add_argument("--results", default="results", help="Directory with .pt models")
    args = ap.parse_args()

    models = load_models(args.results)
    if not models:
        print(f"Error: No trained models found in {args.results}/. Run `python main.py` first.")
        return

    if args.date:
        df = clean(load_csv(args.data)).set_index("date")
        target_date = pd_date = np.datetime64(args.date)
        if target_date not in df.index.values:
            print(f"Date {args.date} not found in {args.data}")
            return
        idx = df.index.get_loc(target_date)
        if idx < 5:
            print("Need at least 5 days of history prior to target date.")
            return

        rain_hist = df["rainfall"].iloc[idx - 5:idx].values[::-1]  # [t-1, t-2, t-3, t-4, t-5]
        runoff_hist = df["runoff"].iloc[idx - 5:idx].values[::-1]
        observed = float(df["runoff"].iloc[idx])
        print(f"\n================ Target Date: {args.date} ================")
        print(f"Past Rainfall (t-1..t-3): {rain_hist[:3].round(2)} inches")
        print(f"Past Runoff   (t-1..t-4): {runoff_hist[:4].round(0)} ft^3/s")
        print(f"Actual Observed Runoff  : {observed:.1f} ft^3/s")
        print("---------------------------------------------------------")
        rain_3, runoff_4 = rain_hist[:3], runoff_hist[:4]
        rain_5, runoff_5 = rain_hist[:5], runoff_hist[:5]
    else:
        rain_hist = args.rain or [0.4, 1.2, 0.1, 0.0, 0.0]
        runoff_hist = args.runoff or [350.0, 480.0, 1100.0, 1600.0, 800.0]
        observed = None
        print("\n================ User Scenario Simulation ================")
        print(f"Rainfall Lags (t-1..t-3): {rain_hist[:3]} inches")
        print(f"Runoff Lags   (t-1..t-4): {runoff_hist[:4]} ft^3/s")
        print("---------------------------------------------------------")
        rain_3 = rain_hist[:3] if len(rain_hist) >= 3 else [0.0]*3
        runoff_4 = runoff_hist[:4] if len(runoff_hist) >= 4 else [100.0]*4
        rain_5 = rain_hist[:5] if len(rain_hist) >= 5 else (rain_hist + [0.0]*5)[:5]
        runoff_5 = runoff_hist[:5] if len(runoff_hist) >= 5 else (runoff_hist + [100.0]*5)[:5]

    for name, model in models.items():
        pred = predict_ff(model, rain_3, runoff_4) if name == "feedforward" else predict_lstm(model, rain_5, runoff_5)
        status = "🚨 FLOOD WARNING (>1500 cfs)" if pred >= PEAK_THRESHOLD else "🟢 Normal Flow"
        err_str = f" | Error: {pred - observed:+.1f} cfs ({abs(pred - observed)/observed*100:.1f}%)" if observed is not None else ""
        print(f"{name.upper():12s} Forecast: {pred:7.1f} ft^3/s | {status}{err_str}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
