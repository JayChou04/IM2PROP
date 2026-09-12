"""MicroNet-based regression architectures for IM2PROP."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from im2prop.config import DEFAULT_CONFIG


def _get_cfg(config: Optional[dict[str, Any]], key: str, default: Any = None) -> Any:
    """Read a value from user config, then DEFAULT_CONFIG, then fallback default."""
    if config is not None and key in config:
        return config[key]
    if key in DEFAULT_CONFIG:
        return DEFAULT_CONFIG[key]
    return default


class PhaseCNN(nn.Module):
    """Extract topological features from a binary phase mask.

    Input shape:
    - [B, 1, H, W]

    Output shape:
    - [B, out_channels, H', W']
    """

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()

        if out_channels != 128:
            raise ValueError("PhaseCNN currently supports out_channels=128 only.")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PhaseAttentionHead(nn.Module):
    """Generate spatial attention from aligned phase features."""

    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()
        self.phase_to_spatial_weight = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, phase_feat_aligned: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.phase_to_spatial_weight(phase_feat_aligned))


class MicroNetRegressor_V2(nn.Module):
    """IM2PROP v2 phase-guided regression model.

    Expected inputs:
    - rgb_img: [B, 3, H, W]
    - phase_mask: [B, 1, H, W]
    - phase_ratios: [B, 2]

    Returns:
    - output: [B, 1]
    - phase_attention_map: [B, 1, H_r, W_r]
    """

    def __init__(
        self,
        pre_trained_base_model: nn.Module,
        *,
        use_phase_attention: Optional[bool] = None,
        use_phase_ratios: Optional[bool] = None,
        use_phase_feat: Optional[bool] = None,
        config: Optional[dict[str, Any]] = None,
        rgb_feat_channels: int = 2048,
        phase_feat_channels: int = 128,
        phase_ratio_dim: int = 2,
        regressor_hidden_dims: tuple[int, int, int, int] = (1024, 512, 256, 128),
    ) -> None:
        super().__init__()

        self.use_phase_attention = bool(
            _get_cfg(config, "USE_PHASE_ATTENTION")
            if use_phase_attention is None
            else use_phase_attention
        )
        self.use_phase_ratios = bool(
            _get_cfg(config, "USE_PHASE_RATIOS")
            if use_phase_ratios is None
            else use_phase_ratios
        )
        self.use_phase_feat = bool(
            _get_cfg(config, "USE_PHASE_FEAT")
            if use_phase_feat is None
            else use_phase_feat
        )

        self.encoder_features = nn.Sequential(
            pre_trained_base_model.conv1,
            pre_trained_base_model.bn1,
            pre_trained_base_model.relu,
            pre_trained_base_model.maxpool,
            pre_trained_base_model.layer1,
            pre_trained_base_model.layer2,
            pre_trained_base_model.layer3,
            pre_trained_base_model.layer4,
        )

        self.phase_cnn = PhaseCNN(out_channels=phase_feat_channels)
        self.phase_attention = PhaseAttentionHead(in_channels=phase_feat_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        reg_in_dim = rgb_feat_channels + phase_feat_channels + phase_ratio_dim
        h1, h2, h3, h4 = regressor_hidden_dims

        self.regressor = nn.Sequential(
            nn.Linear(reg_in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, h4),
            nn.BatchNorm1d(h4),
            nn.ReLU(inplace=True),
            nn.Linear(h4, 1),
        )

    def forward(
        self,
        rgb_img: torch.Tensor,
        phase_mask: torch.Tensor,
        phase_ratios: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_feat = self.encoder_features(rgb_img)
        phase_feat = self.phase_cnn(phase_mask)

        h_r, w_r = rgb_feat.shape[2], rgb_feat.shape[3]
        phase_feat_aligned = F.adaptive_avg_pool2d(phase_feat, (h_r, w_r))

        if self.use_phase_attention:
            phase_attention_map = self.phase_attention(phase_feat_aligned)
        else:
            phase_attention_map = torch.ones(
                (rgb_feat.size(0), 1, h_r, w_r),
                device=rgb_feat.device,
                dtype=rgb_feat.dtype,
            )

        attended_rgb_feat = rgb_feat * phase_attention_map

        rgb_pooled = self.avgpool(attended_rgb_feat)
        rgb_pooled = torch.flatten(rgb_pooled, 1)

        if not self.use_phase_feat:
            phase_feat = torch.zeros_like(phase_feat)

        phase_pooled = self.avgpool(phase_feat)
        phase_pooled = torch.flatten(phase_pooled, 1)

        if not self.use_phase_ratios:
            phase_ratios = torch.zeros_like(phase_ratios)

        combined = torch.cat([rgb_pooled, phase_pooled, phase_ratios], dim=1)
        output = self.regressor(combined)

        return output, phase_attention_map
