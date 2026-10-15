"""Training loop and a single-experiment runner."""
import copy
import numpy as np
import torch
import torch.nn as nn

from .data import prepare
from .metrics import PEAK_THRESHOLD, evaluate
from .models import FeedForward, LSTMNet, to_sequence


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(df, model_type, lag_rain, lag_runoff, hidden, lr,
                   epochs=200, batch_size=64, seed=0, return_model=False,
                   peak_weight=1.0):
    """Train one configuration; keep the epoch with the lowest validation RMSE."""
    set_seed(seed)
    if model_type == "lstm":
        lag_runoff = lag_rain  # LSTM consumes paired (R, Q) steps, so m == n
    ds = prepare(df, lag_rain, lag_runoff)

    if model_type == "lstm":
        Xtr, Xva = to_sequence(ds.X_train, lag_rain), to_sequence(ds.X_val, lag_rain)
        model = LSTMNet(hidden)
    else:
        Xtr = torch.tensor(ds.X_train, dtype=torch.float32)
        Xva = torch.tensor(ds.X_val, dtype=torch.float32)
        model = FeedForward(Xtr.shape[1], hidden)
    ytr = torch.tensor(ds.y_train, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    peak_thresh_norm = PEAK_THRESHOLD / ds.q_max
    best, best_state, best_pred, history = np.inf, None, None, []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            pred_b = model(xb)
            diff_sq = (pred_b - yb) ** 2
            if peak_weight > 1.0:
                weights = torch.where(yb > peak_thresh_norm,
                                      torch.tensor(peak_weight, dtype=torch.float32),
                                      torch.tensor(1.0, dtype=torch.float32))
                loss = torch.mean(weights * diff_sq)
            else:
                loss = torch.mean(diff_sq)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva).numpy()
        val = float(np.sqrt(np.mean((pred - ds.y_val) ** 2)))
        history.append(val)
        if val < best:
            best, best_state, best_pred = val, copy.deepcopy(model.state_dict()), pred

    metrics = evaluate(ds.y_val, best_pred, ds.q_max)
    result = {"model": model_type, "lag_rain": lag_rain, "lag_runoff": lag_runoff,
              "hidden": hidden, "lr": lr, "peak_weight": peak_weight,
              "best_epoch": int(np.argmin(history)) + 1,
              **metrics}
    if return_model:
        model.load_state_dict(best_state)
        return result, model, ds, best_pred, history
    return result
