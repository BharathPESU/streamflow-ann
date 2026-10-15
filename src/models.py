"""Feed-forward network and LSTM for one-step runoff prediction."""
import numpy as np
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """3-layer network: input -> sigmoid hidden -> linear output (as in Hsu et al.)."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.hidden = nn.Linear(n_in, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(torch.sigmoid(self.hidden(x))).squeeze(-1)


class LSTMNet(nn.Module):
    """Input is a sequence of `lag` steps, each [rainfall, runoff] (oldest first)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        seq, _ = self.lstm(x)
        return self.out(seq[:, -1]).squeeze(-1)


def to_sequence(X, lag):
    """Reshape flat [R(t-1..t-lag), Q(t-1..t-lag)] to (N, lag, 2), oldest first."""
    r = X[:, :lag][:, ::-1]
    q = X[:, lag:2 * lag][:, ::-1]
    return torch.tensor(np.stack([r, q], axis=-1).copy(), dtype=torch.float32)
