"""Training engine for IM2PROP v2 models."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def train_one_epoch_v2(model, dataloader, optimizer, criterion, device) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    running_loss = 0.0

    for rgb_tiles, mask_tiles, phase_ratios, targets in tqdm(
        dataloader, desc="Training", leave=False
    ):
        rgb_tiles = rgb_tiles.to(device)
        mask_tiles = mask_tiles.to(device)
        phase_ratios = phase_ratios.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs, _ = model(rgb_tiles, mask_tiles, phase_ratios)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * rgb_tiles.size(0)

    return running_loss / len(dataloader.dataset)


def validate_one_epoch_v2(model, dataloader, criterion, device) -> tuple[float, float, float, float]:
    """Run one validation epoch and return loss, MAE, MSE, RMSE."""
    model.eval()
    running_loss = 0.0
    preds = []
    trues = []

    with torch.no_grad():
        for rgb_tiles, mask_tiles, phase_ratios, targets in tqdm(
            dataloader, desc="Validation", leave=False
        ):
            rgb_tiles = rgb_tiles.to(device)
            mask_tiles = mask_tiles.to(device)
            phase_ratios = phase_ratios.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs, _ = model(rgb_tiles, mask_tiles, phase_ratios)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * rgb_tiles.size(0)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    preds_concat = np.concatenate(preds).squeeze()
    trues_concat = np.concatenate(trues).squeeze()

    mse = float(np.mean((preds_concat - trues_concat) ** 2))
    mae = float(np.mean(np.abs(preds_concat - trues_concat)))
    rmse = float(np.sqrt(mse))

    return epoch_loss, mae, mse, rmse
