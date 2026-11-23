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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        Construct genuine temporal sliding window sequences (N, seq_len, F)
        from chronological feature observations.
        If input is already 3D (N, seq_len, F), returns as is.
        For 2D chronological arrays, each step t contains [x_{t-seq_len+1}, ..., x_t]
        padded with the earliest historical observation for initial warm-up steps.
        """
        if X_scaled.ndim == 3:
            return X_scaled.astype(np.float32)

        N, F = X_scaled.shape
        seqs = np.zeros((N, self.seq_len, F), dtype=np.float32)
        for i in range(N):
            start = max(0, i - self.seq_len + 1)
            window = X_scaled[start : i + 1]
            if len(window) < self.seq_len:
                pad_len = self.seq_len - len(window)
                pad = np.repeat(window[:1], pad_len, axis=0)
                window = np.vstack([pad, window])
            seqs[i] = window
        return seqs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        """
        Fit scaler and LSTM network on training fold.
        """
        set_seed(self.random_state)
        if X.ndim == 3:
            # Flatten to scale features across all sequence steps then restore shape
            N, S, F = X.shape
            X_flat = X.reshape(-1, F)
            X_scaled_flat = self.scaler.fit_transform(X_flat)
            X_seq = X_scaled_flat.reshape(N, S, F).astype(np.float32)
        else:
            X_scaled = self.scaler.fit_transform(X)
            X_seq = self._prepare_sequences(X_scaled)

        tensor_x = torch.tensor(X_seq, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)

        self.net = PyTorchLSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1
        )
        self.net.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        N = len(tensor_x)
        batch_size = min(self.batch_size, N)

        for epoch in range(self.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                batch_x = tensor_x[idx]
                batch_y = tensor_y[idx]

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
        """
        if self.net is None:
            raise RuntimeError("Model must be fitted before predict.")

        self.net.eval()
        if X.ndim == 3:
            N, S, F = X.shape
            X_flat = X.reshape(-1, F)
            X_scaled_flat = self.scaler.transform(X_flat)
            X_seq = X_scaled_flat.reshape(N, S, F).astype(np.float32)
        else:
            X_scaled = self.scaler.transform(X)
            X_seq = self._prepare_sequences(X_scaled)

        tensor_x = torch.tensor(X_seq, dtype=torch.float32)

        with torch.no_grad():
            preds = self.net(tensor_x).cpu().numpy()
        return preds
