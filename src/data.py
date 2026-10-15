"""Data loading, cleaning, lag-window construction and normalisation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def make_synthetic(seed: int = 0) -> pd.DataFrame:
    """Synthetic daily rainfall/runoff series (2003-2012) for testing the pipeline.

    NOT real data. Rainfall is a gamma storm process; runoff comes from a
    non-linear reservoir, scaled so the peak is ~29,700 cfs like the Leaf River.
    Use only to verify the code runs end to end.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2003-01-01", "2012-12-31", freq="D")
    n = len(dates)
    wet = rng.random(n) < 0.28
    rain = np.where(wet, rng.gamma(0.8, 1.1, n), 0.0)
    rain = np.clip(rain, 0, 6.8)
    store, q = 0.0, np.zeros(n)
    for t in range(n):
        store += rain[t] * 0.6
        out = 0.09 * store + 0.12 * store ** 1.6 / (1 + store)
        out = min(out, store)
        store -= out
        q[t] = out
    q = (q / q.max()) ** 2.2 * 29700 + rng.normal(0, 15, n).clip(-30, 30) + 80
    return pd.DataFrame({"date": dates, "rainfall": rain, "runoff": q.clip(min=0)})


def load_csv(path: str) -> pd.DataFrame:
    """CSV with columns: date, rainfall (inch), runoff (ft^3/s)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "rainfall", "runoff"]].copy()
    df[["rainfall", "runoff"]] = df[["rainfall", "runoff"]].apply(
        pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a full daily calendar; interpolate short runoff gaps only.

    Runoff is smooth, so gaps of up to 3 days are linearly interpolated.
    Rainfall is intermittent, so it is never interpolated: any lag window that
    still contains a missing value is dropped later.
    """
    df = df.drop_duplicates("date").set_index("date")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df["runoff"] = df["runoff"].interpolate(limit=3, limit_area="inside")
    df.index.name = "date"
    return df.reset_index()


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    q_max: float
    r_max: float
    lag_rain: int
    lag_runoff: int


def build_windows(rain, runoff, lag_rain, lag_runoff):
    """Targets Q(t); features R(t-1..t-lag_rain), Q(t-1..t-lag_runoff)."""
    start = max(lag_rain, lag_runoff)
    X, y = [], []
    for t in range(start, len(runoff)):
        X.append(np.concatenate([rain[t - lag_rain:t][::-1],
                                 runoff[t - lag_runoff:t][::-1]]))
        y.append(runoff[t])
    X, y = np.array(X), np.array(y)
    ok = ~(np.isnan(X).any(1) | np.isnan(y))
    return X, y, ok, start


def prepare(df, lag_rain, lag_runoff, train_years=(2003, 2007),
            val_years=(2008, 2012)) -> Dataset:
    """Chronological split by year, then scale to [0, 1].

    Scale factors are the training-set maxima (avoids leaking validation
    information). Validation values above the training max may exceed 1,
    which is fine for a linear output layer.
    """
    df = clean(df)
    years = df["date"].dt.year.values
    rain, runoff = df["rainfall"].values, df["runoff"].values
    X, y, ok, start = build_windows(rain, runoff, lag_rain, lag_runoff)
    tgt_years = years[start:]
    tr = ok & (tgt_years >= train_years[0]) & (tgt_years <= train_years[1])
    va = ok & (tgt_years >= val_years[0]) & (tgt_years <= val_years[1])

    r_max = np.nanmax(rain[years <= train_years[1]])
    q_max = np.nanmax(runoff[years <= train_years[1]])
    scale = np.array([r_max] * lag_rain + [q_max] * lag_runoff)
    return Dataset(X[tr] / scale, y[tr] / q_max, X[va] / scale, y[va] / q_max,
                   q_max, r_max, lag_rain, lag_runoff)
