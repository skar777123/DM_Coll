"""
ML_Model/models/threat_lstm.py
────────────────────────────────
LSTM-based Threat Predictor for BlindSpotGuard.

Architecture
────────────
  Input  : [batch, seq_len, 15 features]  (3 sensors × 5 features each)
  LSTM   : 2 layers, hidden=128, bidirectional=False
  Head   :
    ├─ ClassHead   → 3-class softmax (safe / caution / critical)
    └─ TTCHead     → 1 scalar regression (time-to-collision, seconds)

Outputs
───────
  logits  : [batch, 3]     → cross-entropy loss
  ttc_pred: [batch, 1]     → MSE loss (log-space for stability)
  probs   : [batch, 3]     → softmax probabilities (inference)

Design Choices
──────────────
• Shared LSTM encoder → two separate heads (multi-task learning)
• Dropout on LSTM + FC for regularisation on small Pi-side data
• TTLogprobability only at last timestep (we want the final verdict)
• Lightweight enough to run on Pi for real-time (≈ 2 ms / forward pass)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_Model.config_ml import LSTM as LSTM_CFG, INPUT_FEATURES, SEQUENCE_LEN, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────

class _LSTMEncoder(nn.Module):
    """Shared LSTM backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=LSTM_CFG["hidden_size"],
            num_layers=LSTM_CFG["num_layers"],
            batch_first=True,
            dropout=LSTM_CFG["dropout"] if LSTM_CFG["num_layers"] > 1 else 0,
            bidirectional=LSTM_CFG["bidirectional"],
        )
        self.layer_norm = nn.LayerNorm(LSTM_CFG["hidden_size"])
        self.dropout    = nn.Dropout(LSTM_CFG["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            last hidden state: [batch, hidden_size]
        """
        out, _ = self.lstm(x)          # [batch, seq, hidden]
        last   = out[:, -1, :]         # last timestep
        last   = self.layer_norm(last)
        return self.dropout(last)


class _ClassHead(nn.Module):
    """Multi-class zone classification head."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # raw logits


class _TTCHead(nn.Module):
    """Time-to-collision regression head."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),   # TTC must be ≥ 0
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # [batch, 1]


# ─────────────────────────────────────────────────────────────────────────────

class ThreatLSTM(nn.Module):
    """
    Full threat prediction model.

    Usage (training)::

        model   = ThreatLSTM()
        logits, ttc_pred = model(x)
        loss    = model.compute_loss(logits, ttc_pred, y_labels, y_ttc)

    Usage (inference)::

        with torch.no_grad():
            probs, ttc = model.predict(x)
        zone = model.zone_from_probs(probs)
    """

    def __init__(self) -> None:
        super().__init__()
        hidden = LSTM_CFG["hidden_size"]
        self.encoder   = _LSTMEncoder()
        self.cls_head  = _ClassHead(hidden)
        self.ttc_head  = _TTCHead(hidden)

        # Learnable loss weight (log-scale for numerical stability)
        self.log_alpha = nn.Parameter(torch.zeros(1))   # cls weight
        self.log_beta  = nn.Parameter(torch.zeros(1))   # ttc weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z        = self.encoder(x)
        logits   = self.cls_head(z)
        ttc_pred = self.ttc_head(z)
        return logits, ttc_pred

    def compute_loss(
        self,
        logits:   torch.Tensor,
        ttc_pred: torch.Tensor,
        y_cls:    torch.Tensor,
        y_ttc:    torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-task loss = α·CrossEntropy + β·HuberLoss(log TTC).

        Weights are learnable (Uncertainty Weighting, Kendall et al. 2018).
        Returns: (total_loss, cls_loss, ttc_loss)
        """
        # Classification
        cls_loss = F.cross_entropy(logits, y_cls, weight=class_weights)

        # TTC regression in log-space (log(TTC+1) for stability)
        log_ttc_pred = torch.log1p(ttc_pred.squeeze(1))
        log_ttc_true = torch.log1p(y_ttc.clamp(0, 30))
        ttc_loss = F.huber_loss(log_ttc_pred, log_ttc_true, delta=1.0)

        # Uncertainty-weighted combination
        alpha    = torch.exp(-self.log_alpha)
        beta     = torch.exp(-self.log_beta)
        total    = alpha * cls_loss + self.log_alpha + \
                   beta  * ttc_loss + self.log_beta
        return total, cls_loss, ttc_loss

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs    : [batch, 3]  — class probabilities
            ttc_pred : [batch]     — TTC in seconds
        """
        logits, ttc_raw = self(x)
        probs = F.softmax(logits, dim=-1)
        ttc   = ttc_raw.squeeze(1)
        return probs, ttc

    @staticmethod
    def zone_from_probs(probs: torch.Tensor) -> list[str]:
        """Convert probability tensor → zone label strings."""
        from ML_Model.config_ml import CLASSES, THRESHOLDS
        results = []
        for p in probs:
            crit_p = p[2].item()
            caut_p = p[1].item()
            if crit_p >= THRESHOLDS["critical_prob"]:
                results.append("critical")
            elif crit_p >= THRESHOLDS["caution_prob"] or caut_p >= THRESHOLDS["caution_prob"]:
                results.append("caution")
            else:
                results.append("safe")
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
