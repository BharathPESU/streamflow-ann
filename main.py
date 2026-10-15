"""Train and evaluate the feed-forward network and LSTM with chosen hyperparameters.

Examples
    python main.py --data data/leaf_river.csv
    python main.py --synthetic --epochs 30          # pipeline check only
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import load_csv, make_synthetic, prepare
from src.metrics import evaluate
from src.train import run_experiment


def get_df(args):
    if args.synthetic:
        print("WARNING: using SYNTHETIC data - results are not meaningful.")
        return make_synthetic()
    return load_csv(args.data)


def plot_prediction(name, ds, pred, out_dir):
    t, p = ds.y_val * ds.q_max, pred * ds.q_max
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    ax[0].plot(t, "b--", lw=0.8, label="observed")
    ax[0].plot(p, "g", lw=0.8, label="predicted")
    ax[0].axhline(1500, color="r", ls=":", lw=0.8, alpha=0.7, label="peak threshold (1500)")
    ax[0].set(title=f"{name}: validation series", xlabel="day index",
              ylabel="runoff (ft$^3$/s)")
    ax[0].legend()
    ax[1].scatter(t, p, s=6, alpha=0.5)
    m = max(t.max(), p.max())
    ax[1].plot([0, m], [0, m], "k:", lw=1)
    ax[1].axvline(1500, color="r", ls=":", lw=0.8, alpha=0.5)
    ax[1].axhline(1500, color="r", ls=":", lw=0.8, alpha=0.5)
    ax[1].set(title="observed vs predicted", xlabel="observed", ylabel="predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pred_{name}.png"), dpi=150)
    plt.close(fig)


def plot_comparison(ds, pred_ff, pred_lstm, out_dir):
    """Side-by-side comparative hydrograph and storm-event zoom for presentations."""
    t = ds.y_val * ds.q_max
    p_ff = pred_ff * ds.q_max
    p_lstm = pred_lstm * ds.q_max

    # 1. Full comparison hydrograph
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, "k-", lw=0.8, alpha=0.8, label="Observed Runoff")
    ax.plot(p_ff, "b--", lw=0.9, alpha=0.85, label="Feed-Forward ANN")
    ax.plot(p_lstm, "r-.", lw=0.9, alpha=0.85, label="LSTM Network")
    ax.axhline(1500, color="orange", ls=":", lw=1, label="Flood Threshold (1500 ft³/s)")
    ax.set(title="Model Comparison: Observed vs. Feed-Forward vs. LSTM (Validation 2008-2012)",
           xlabel="Day Index", ylabel="Runoff (ft$^3$/s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Zoomed-in major flood event (highest peak in validation set)
    max_idx = int(np.argmax(t))
    w_start = max(0, max_idx - 30)
    w_end = min(len(t), max_idx + 30)
    days = np.arange(w_start, w_end)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(days, t[w_start:w_end], "k-o", ms=3, lw=1.2, label="Observed Runoff")
    ax.plot(days, p_ff[w_start:w_end], "b--s", ms=2.5, lw=1.0, label="Feed-Forward")
    ax.plot(days, p_lstm[w_start:w_end], "r-.^", ms=2.5, lw=1.0, label="LSTM")
    ax.axhline(1500, color="orange", ls=":", lw=1, label="Flood Peak Threshold")
    ax.set(title=f"Storm Event Zoom-In: Peak Flood Hydrograph (Days {w_start} to {w_end})",
           xlabel="Day Index", ylabel="Runoff (ft$^3$/s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "peak_event_zoom.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/leaf_river.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-weight", type=float, default=1.0,
                    help="Loss weight multiplier for flood peaks Q > 1500 (default: 1.0)")
    # defaults = best configs reported in the reference study; re-tune with sweep.py
    ap.add_argument("--ff", nargs=4, type=float, default=[3, 4, 12, 0.009],
                    metavar=("LAG_RAIN", "LAG_RUNOFF", "HIDDEN", "LR"))
    ap.add_argument("--lstm", nargs=3, type=float, default=[5, 12, 0.001],
                    metavar=("LAG", "HIDDEN", "LR"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = get_df(args)

    results = {}
    lr_, lo_, h, lr = args.ff
    res_ff, model_ff, ds_ff, pred_ff, _ = run_experiment(
        df, "ff", int(lr_), int(lo_), int(h), lr, epochs=args.epochs,
        seed=args.seed, return_model=True, peak_weight=args.peak_weight)
    results["feedforward"] = res_ff
    plot_prediction("feedforward", ds_ff, pred_ff, args.out)
    ff_model_path = os.path.join(args.out, "model_feedforward.pt")
    torch.save(model_ff.state_dict(), ff_model_path)

    lag, h, lr = args.lstm
    res_lstm, model_lstm, ds_lstm, pred_lstm, _ = run_experiment(
        df, "lstm", int(lag), int(lag), int(h), lr, epochs=args.epochs,
        seed=args.seed, return_model=True, peak_weight=args.peak_weight)
    results["lstm"] = res_lstm
    plot_prediction("lstm", ds_lstm, pred_lstm, args.out)
    lstm_model_path = os.path.join(args.out, "model_lstm.pt")
    torch.save(model_lstm.state_dict(), lstm_model_path)

    # Multi-model comparative plots for presentations
    plot_comparison(ds_ff, pred_ff, pred_lstm, args.out)

    # naive baseline: tomorrow's runoff = today's runoff
    bds = prepare(df, int(lr_), int(lo_))
    results["persistence"] = evaluate(bds.y_val, bds.X_val[:, int(lr_)], bds.q_max)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved models to {args.out}/:")
    print(f"  - {ff_model_path}")
    print(f"  - {lstm_model_path}")


    print(f"\n{'model':14s}{'overall%':>9s}{'peak%':>8s}{'nonpeak%':>9s}{'NSE':>8s}{'KGE':>8s}")
    for k, r in results.items():
        nse_val = r.get("nse_overall", float("nan"))
        kge_val = r.get("kge_overall", float("nan"))
        print(f"{k:14s}{r['overall_pct']:9.2f}{r['peak_pct']:8.2f}{r['nonpeak_pct']:9.2f}{nse_val:8.3f}{kge_val:8.3f}")



if __name__ == "__main__":
    main()
