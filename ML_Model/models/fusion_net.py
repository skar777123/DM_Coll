"""
ML_Model/models/fusion_net.py
──────────────────────────────
Sensor Fusion Neural Network for BlindSpotGuard.

Takes a SINGLE timestep snapshot from all sensors (no sequence — instant
classification). Designed as a fast, lightweight fallback when the
LSTM inference latency is too high (e.g., Pi under load).

Architecture
────────────
  Input  : [batch, 15] — one timestep of 3 sensors × 5 features
            [dist_l, vel_l, acc_l, zone_l, cam_l,
             dist_r, vel_r, acc_r, zone_r, cam_r,
             dist_b, vel_b, acc_b, zone_b, cam_b]

  Sensor branch × 3 (weight-shared):
    Linear(5→32) → BN → GELU → Dropout

  Fusion:
    Concat(96) → Linear(128) → BN → GELU
             → Linear(64)  → BN → GELU
             → ClassHead(3)
             → TTCHead(1)

Inference: ≈ 0.1 ms on Pi — suitable for 50 Hz loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_Model.config_ml import FUSION_NET, SENSOR_FEATURES, N_SENSORS, NUM_CLASSES, THRESHOLDS


# ─────────────────────────────────────────────────────────────────────────────

class _SensorBranch(nn.Module):
    """Shared-weight encoder for one sensor's 5-feature vector."""

    def __init__(self, in_dim: int = SENSOR_FEATURES, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionNet(nn.Module):
    """
    Fast single-timestep sensor fusion classifier.

    Usage (training)::

        model  = FusionNet()
        logits, ttc = model(x)    # x: [batch, 15]
        loss = model.compute_loss(logits, ttc, y_cls, y_ttc)

    Usage (inference)::

        with torch.no_grad():
            probs, ttc = model.predict(x_snapshot)
    """

    def __init__(self) -> None:
        super().__init__()
        branch_out = 32
        fusion_in  = branch_out * N_SENSORS    # 96

        # Weight-shared sensor branches
        self.sensor_branch = _SensorBranch(SENSOR_FEATURES, branch_out)

        # Fusion MLP
        dims = FUSION_NET["hidden_dims"]   # [128, 64, 32]
        layers = []
        in_d   = fusion_in
        for d in dims:
            layers += [nn.Linear(in_d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(FUSION_NET["dropout"])]
            in_d = d
        self.fusion = nn.Sequential(*layers)

        # Heads
        self.cls_head = nn.Linear(in_d, NUM_CLASSES)
        self.ttc_head = nn.Sequential(
            nn.Linear(in_d, 16), nn.GELU(), nn.Linear(16, 1), nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 15]   — concatenated sensor features
        Returns:
            logits  : [batch, 3]
            ttc_pred: [batch, 1]
        """
        # Split into per-sensor chunks and encode
        encoded = []
        for i in range(N_SENSORS):
            chunk = x[:, i * SENSOR_FEATURES : (i + 1) * SENSOR_FEATURES]
            encoded.append(self.sensor_branch(chunk))

        fused    = torch.cat(encoded, dim=1)   # [batch, 96]
        fused    = self.fusion(fused)           # [batch, 32]
        logits   = self.cls_head(fused)
        ttc_pred = self.ttc_head(fused)
        return logits, ttc_pred

    def compute_loss(
        self,
        logits:   torch.Tensor,
        ttc_pred: torch.Tensor,
        y_cls:    torch.Tensor,
        y_ttc:    torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = F.cross_entropy(logits, y_cls, weight=class_weights)
        log_ttc_pred = torch.log1p(ttc_pred.squeeze(1))
        log_ttc_true = torch.log1p(y_ttc.clamp(0, 30))
        ttc_loss     = F.huber_loss(log_ttc_pred, log_ttc_true, delta=1.0)
        total        = cls_loss + 0.3 * ttc_loss
        return total, cls_loss, ttc_loss

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, ttc = self(x)
        probs = F.softmax(logits, dim=-1)
        return probs, ttc.squeeze(1)

    @staticmethod
    def zone_from_probs(probs: torch.Tensor) -> list[str]:
        results = []
        for p in probs:
            if p[2].item() >= THRESHOLDS["critical_prob"]:
                results.append("critical")
            elif p[1].item() + p[2].item() >= THRESHOLDS["caution_prob"]:
                results.append("caution")
            else:
                results.append("safe")
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
