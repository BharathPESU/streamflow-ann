"""RMSE overall / at peaks / at non-peaks (threshold on raw runoff, ft^3/s)."""
import numpy as np

PEAK_THRESHOLD = 1500.0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency: 1 - sum((y - y_hat)^2) / sum((y - y_mean)^2)."""
    if len(y_true) < 2:
        return float("nan")
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def kge(y_true, y_pred):
    """Kling-Gupta Efficiency: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    if len(y_true) < 2:
        return float("nan")
    std_true, std_pred = np.std(y_true), np.std(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    if std_true == 0 or mean_true == 0:
        return float("nan")
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if std_pred > 0 else 0.0
    alpha = float(std_pred / std_true)
    beta = float(mean_pred / mean_true)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def evaluate(y_true, y_pred, q_max):
    """y_* are normalised; returns raw-unit RMSEs, percentages, NSE, and KGE."""
    t, p = y_true * q_max, y_pred * q_max
    peak = t > PEAK_THRESHOLD
    out = {
        "overall": rmse(t, p),
        "peak": rmse(t[peak], p[peak]),
        "nonpeak": rmse(t[~peak], p[~peak]),
        "n_peak": int(peak.sum()),
        "nse_overall": nse(t, p),
        "nse_peak": nse(t[peak], p[peak]),
        "nse_nonpeak": nse(t[~peak], p[~peak]),
        "kge_overall": kge(t, p),
    }
    out.update({f"{k}_pct": 100 * out[k] / q_max for k in ("overall", "peak", "nonpeak")})
    return out

