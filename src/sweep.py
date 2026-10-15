"""Grid search over lag days and hidden size (learning rate fixed per model).

    python sweep.py --data data/leaf_river.csv --model ff
    python sweep.py --synthetic --model lstm --epochs 20
"""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_csv, make_synthetic
from src.train import run_experiment



def heatmap(df, metric, title, path):
    piv = df.pivot_table(index="lag_rain", columns="hidden", values=metric)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="coolwarm_r")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set(xlabel="hidden size", ylabel="rainfall lag days", title=title)
    fig.colorbar(im, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/leaf_river.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", choices=["ff", "lstm"], default="ff")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lags", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--hidden", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--runoff-extra", nargs="+", type=int, default=[1, 2, 3],
                    help="runoff lag minus rainfall lag (ff only)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = make_synthetic() if args.synthetic else load_csv(args.data)
    lr = args.lr or (0.009 if args.model == "ff" else 0.001)

    extras = args.runoff_extra if args.model == "ff" else [0]
    rows = []
    for lag, extra, hid in itertools.product(args.lags, extras, args.hidden):
        r = run_experiment(df, args.model, lag, lag + extra, hid, lr, epochs=args.epochs)
        rows.append(r)
        print(f"lag_rain={lag} lag_runoff={r['lag_runoff']} hidden={hid} "
              f"overall={r['overall_pct']:.3f}%", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, f"sweep_{args.model}.csv"), index=False)
    best = res.loc[res["overall"].idxmin()]
    print("\nBEST by overall RMSE:\n", best.to_string())
    for extra, g in res.groupby(res["lag_runoff"] - res["lag_rain"]):
        for metric in ("overall_pct", "peak_pct", "nonpeak_pct"):
            heatmap(g, metric, f"{args.model} n-m={extra} {metric}",
                    os.path.join(args.out, f"heat_{args.model}_d{extra}_{metric}.png"))


if __name__ == "__main__":
    main()
