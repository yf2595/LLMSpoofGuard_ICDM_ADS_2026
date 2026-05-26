"""
LSTM baseline (paper Table II, S; T).

Supervised, trajectory-level. A small bidirectional LSTM ingests the
full padded ADS-B sequence and predicts a single binary spoofing label
per trajectory. Operates on ``src.features.sequence_tensor`` output and
uses a mean-pool over the masked time dimension for classification.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import SEQUENCE_FEATURE_COLS, compute_norm_stats, sequence_tensor

from .base import Baseline


class _LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


class LSTMBaseline(Baseline):
    name = "LSTM"
    is_supervised = True
    granularity = "trajectory"

    def __init__(
        self,
        max_len: int = 64,
        hidden: int = 64,
        num_layers: int = 1,
        epochs: int = 8,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        threshold: float = 0.5,
        device: str | None = None,
        seed: int = 42,
    ):
        self.max_len = max_len
        self.hidden = hidden
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self._fitted = False
        self.model: _LSTMClassifier | None = None
        self.norm_stats: dict | None = None

    def _build_arrays(
        self,
        trajs: Sequence[Sequence[Sequence]],
        labels: Sequence[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feats: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for t in trajs:
            f, m = sequence_tensor(t, self.max_len, self.norm_stats)
            feats.append(f)
            masks.append(m)
        x = torch.from_numpy(np.stack(feats, axis=0)).float() if feats else torch.zeros(0, self.max_len, len(SEQUENCE_FEATURE_COLS))
        msk = torch.from_numpy(np.stack(masks, axis=0)).float() if masks else torch.zeros(0, self.max_len)
        y = torch.tensor([int(bool(v)) for v in labels], dtype=torch.float32) if labels is not None else None
        return x, msk, y

    def fit(
        self,
        train_trajectories: Sequence[Sequence[Sequence]],
        train_labels: Sequence[bool] | None = None,
    ) -> None:
        if train_labels is None:
            raise ValueError("LSTM requires training labels.")
        if not train_trajectories or len(set(bool(v) for v in train_labels)) < 2:
            self._fitted = False
            return

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.norm_stats = compute_norm_stats(train_trajectories, self.max_len)
        x, m, y = self._build_arrays(train_trajectories, train_labels)
        x, m, y = x.to(self.device), m.to(self.device), y.to(self.device)

        n_features = len(SEQUENCE_FEATURE_COLS)
        self.model = _LSTMClassifier(n_features, self.hidden, self.num_layers).to(self.device)

        pos = float(y.sum().item())
        neg = float(y.numel() - pos)
        pos_weight = torch.tensor([neg / max(1.0, pos)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(x, m, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, mb, yb in loader:
                optim.zero_grad()
                logits = self.model(xb, mb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

        self._fitted = True

    def predict(self, test_trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
        if not self._fitted or self.model is None:
            return [False] * len(test_trajectories)
        x, m, _ = self._build_arrays(test_trajectories)
        x, m = x.to(self.device), m.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, m)
            probs = torch.sigmoid(logits).cpu().numpy()
        return [bool(p >= self.threshold) for p in probs]
