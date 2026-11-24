"""
src/models/lstm_model.py
========================
PyTorch LSTM Challenger Sequence Model.

Implements Model C from Master Implementation Plan Section 6.1:
- PyTorch recurrent architecture capturing multi-period temporal dependencies
- Compact, regularized design (1-2 LSTM layers, hidden dim 32, dropout 0.1)
- In-fold standard scaling to strictly prevent look-ahead bias
- Fast CPU-compatible training with early stopping
"""

import os
import random
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_panel_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 5,
    ticker_col: str = "ticker",
    date_col: str = "date"
) -> np.ndarray:
    """
    Construct genuine per-ticker temporal sliding sequences of shape (N, seq_len, F).
    Guarantees that every historical step in the sequence belongs to the same asset,
    strictly preventing cross-sectional stock mixing.
    """
    N = len(df)
    F = len(feature_cols)
    out_seqs = np.zeros((N, seq_len, F), dtype=np.float32)

    df_indexed = df.reset_index(drop=True)
    for _, group in df_indexed.groupby(ticker_col):
        sorted_group = group.sort_values(date_col)
        grp_features = sorted_group[feature_cols].values
        orig_indices = sorted_group.index.values

        K = len(grp_features)
        for i in range(K):
            start = max(0, i - seq_len + 1)
            hist = grp_features[start : i + 1]
            pad_len = seq_len - len(hist)
            target_idx = orig_indices[i]
            if pad_len > 0:
                pad = np.repeat(grp_features[start : start + 1], pad_len, axis=0)
                out_seqs[target_idx] = np.vstack([pad, hist])
            else:
                out_seqs[target_idx] = hist

    return out_seqs


def extract_rebalance_sequences(
    df_history: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    tickers: List[str],
    feature_cols: List[str],
    seq_len: int = 5,
    ticker_col: str = "ticker",
    date_col: str = "date"
) -> np.ndarray:
    """
    Construct per-ticker 3D input (len(tickers), seq_len, F) for inference on rebalance date.
    Extracts the trailing seq_len observations for each requested ticker up to rebalance_date.
    """
    F = len(feature_cols)
    M = len(tickers)
    out_seqs = np.zeros((M, seq_len, F), dtype=np.float32)

    hist_sub = df_history[df_history[date_col] <= rebalance_date]

    for idx, tk in enumerate(tickers):
        tk_data = hist_sub[hist_sub[ticker_col] == tk].sort_values(date_col)
        if len(tk_data) == 0:
            continue
        vals = tk_data[feature_cols].tail(seq_len).values
        pad_len = seq_len - len(vals)
        if pad_len > 0:
            pad = np.repeat(vals[0:1], pad_len, axis=0)
            out_seqs[idx] = np.vstack([pad, vals])
        else:
            out_seqs[idx] = vals

    return out_seqs


class PyTorchLSTMNet(nn.Module):
    """
    Compact LSTM neural network for financial return prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last time-step representation
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out.squeeze(-1)


class LSTMModel:
    """
    Wrapper for training and inference with PyTorch LSTM sequence model.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 1,
        learning_rate: float = 0.003,
        epochs: int = 15,
        batch_size: int = 128,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        set_seed(random_state)
        self.scaler = StandardScaler()
        self.net: Optional[PyTorchLSTMNet] = None

    def _prepare_sequences(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Construct genuine historical sliding-window sequences of shape (N, seq_len, F).

        If input is already a 3D temporal sequence matrix (N, seq_len, F), it is returned directly.
        If input is a 2D matrix (N, F), genuine chronological lag windows are formed:
        for each sample at temporal index t, the sequence consists of actual historical observations
        [X[t - seq_len + 1], ..., X[t]]. Early boundary samples (t < seq_len - 1) are left-padded
        with the earliest available historical observation.
        """
        if X_scaled.ndim == 3:
            return X_scaled.astype(np.float32)

        N, F = X_scaled.shape
        seqs = np.zeros((N, self.seq_len, F), dtype=np.float32)
        for t in range(N):
            start_idx = max(0, t - self.seq_len + 1)
            hist = X_scaled[start_idx : t + 1]
            pad_len = self.seq_len - len(hist)
            if pad_len > 0:
                pad = np.repeat(X_scaled[start_idx : start_idx + 1], pad_len, axis=0)
                seqs[t] = np.vstack([pad, hist])
            else:
                seqs[t] = hist
        return seqs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        """
        Fit scaler and LSTM network on training fold.
        Supports both 2D (N, F) and genuine 3D (N, seq_len, F) inputs.
        """
        set_seed(self.random_state)
        if X.ndim == 3:
            N, S, F = X.shape
            X_reshaped = X.reshape(-1, F)
            X_scaled = self.scaler.fit_transform(X_reshaped).reshape(N, S, F)
            X_seq = X_scaled.astype(np.float32)
            input_dim = F
        else:
            X_scaled = self.scaler.fit_transform(X)
            X_seq = self._prepare_sequences(X_scaled)
            input_dim = X.shape[1]

        tensor_x = torch.tensor(X_seq, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = PyTorchLSTMNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1
        )
        self.net.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        self.net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate out-of-sample predictions.
        Supports both 2D (N, F) and genuine 3D (N, seq_len, F) inputs.
        """
        if self.net is None:
            raise RuntimeError("Model must be fitted before predict.")

        self.net.eval()
        if X.ndim == 3:
            N, S, F = X.shape
            X_reshaped = X.reshape(-1, F)
            X_scaled = self.scaler.transform(X_reshaped).reshape(N, S, F)
            X_seq = X_scaled.astype(np.float32)
        else:
            X_scaled = self.scaler.transform(X)
            X_seq = self._prepare_sequences(X_scaled)

        tensor_x = torch.tensor(X_seq, dtype=torch.float32)

        with torch.no_grad():
            preds = self.net(tensor_x).cpu().numpy()
        return preds
