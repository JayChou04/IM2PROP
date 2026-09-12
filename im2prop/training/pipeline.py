"""Unified train/test pipeline helpers for IM2PROP."""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretrained_microscopy_models as pmm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import wandb
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from im2prop.data.dataset import IM2PROPDataset_V2
from im2prop.models.micronet import MicroNetRegressor_V2
from im2prop.training.engine import train_one_epoch_v2, validate_one_epoch_v2


def print_section(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(title)
    print(line)


def set_global_seed(seed: int) -> None:
    """Pair every ablation combo on one seed: same split, init and batch order."""
    # cudnn determinism left off: costs speed, and the residual noise hits all
    # combos equally so it averages out across repetitions.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_criterion(name: str) -> nn.Module:
    name = name.strip()
    if name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    if name == "MSELoss":
        return nn.MSELoss()
    if name == "L1Loss":
        return nn.L1Loss()
    raise ValueError(f"Unsupported CRITERION: {name}")


def freeze_encoder_if_needed(model: nn.Module, freeze_encoder: bool) -> None:
    if not freeze_encoder:
        return
    for name, param in model.named_parameters():
        if name.startswith("encoder_features"):
            param.requires_grad = False


def build_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    base_model = torch.hub.load(
        "pytorch/vision:v0.10.0",
        cfg["ENCODER_NAME"],
        weights=None,
    )
    url = pmm.util.get_pretrained_microscopynet_url(
        encoder=cfg["ENCODER_NAME"],
        encoder_weights=cfg["WEIGHTS_SOURCE"],
    )
    base_model.load_state_dict(model_zoo.load_url(url, map_location=device))

    model = MicroNetRegressor_V2(
        pre_trained_base_model=base_model,
        config=cfg,
    ).to(device)
    return model


def create_split_indices(df: pd.DataFrame, random_state: int) -> dict[str, list[int]]:
    all_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.15,
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,
        random_state=random_state,
    )
    return {
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
        "random_state": int(random_state),
    }


def create_kfold_split_indices(df: pd.DataFrame, n_splits: int, fold_idx: int, random_state: int) -> dict[str, Any]:
    """Fold `fold_idx` is the test set; the rest splits into train/val."""
    if not (0 <= fold_idx < n_splits):
        raise ValueError(f"fold_idx {fold_idx} out of range for n_splits {n_splits}")
    all_idx = np.arange(len(df))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_val_idx, test_idx = list(kfold.split(all_idx))[fold_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,
        random_state=random_state,
    )
    return {
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
        "random_state": int(random_state),
        "cv_folds": int(n_splits),
        "cv_fold_idx": int(fold_idx),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_splits_from_indices(
    df: pd.DataFrame,
    split_meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[split_meta["train_indices"]].reset_index(drop=True)
    val_df = df.iloc[split_meta["val_indices"]].reset_index(drop=True)
    test_df = df.iloc[split_meta["test_indices"]].reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate_test_set(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, trues = [], []
    rgb_imgs, mask_tiles, attn_maps, ratio_values = [], [], [], []

    with torch.no_grad():
        for rgb_tiles, masks, phase_ratios, targets in dataloader:
            rgb_tiles = rgb_tiles.to(device)
            masks = masks.to(device)
            phase_ratios = phase_ratios.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs, phase_attn = model(rgb_tiles, masks, phase_ratios)

            preds.append(outputs.cpu().numpy())
            trues.append(targets.cpu().numpy())
            rgb_imgs.append(rgb_tiles.cpu().numpy())
            mask_tiles.append(masks.cpu().numpy())
            attn_maps.append(phase_attn.cpu().numpy())
            ratio_values.append(phase_ratios.cpu().numpy())

    preds_concat = np.concatenate(preds).reshape(-1)
    trues_concat = np.concatenate(trues).reshape(-1)
    imgs_concat = np.concatenate(rgb_imgs)
    masks_concat = np.concatenate(mask_tiles)
    attn_concat = np.concatenate(attn_maps)
    ratios_concat = np.concatenate(ratio_values)

    mse = float(np.mean((preds_concat - trues_concat) ** 2))
    mae = float(np.mean(np.abs(preds_concat - trues_concat)))
    rmse = float(np.sqrt(mse))

    return {
        "preds": preds_concat,
        "trues": trues_concat,
        "imgs": imgs_concat,
        "masks": masks_concat,
        "attn": attn_concat,
        "ratios": ratios_concat,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


def visualize_predictions(
    results: dict[str, Any],
    output_path: Path,
    num_samples: int,
    heatmap_cmap: str,
    attention_alpha: float,
    random_state: int,
) -> None:
    imgs = results["imgs"]
    trues = results["trues"]
    preds = results["preds"]
    masks = results["masks"]
    attention_maps = results["attn"]

    sample_count = min(num_samples, len(imgs))
    if sample_count < 1:
        raise ValueError("No samples available for visualization.")

    rng = np.random.default_rng(random_state)
    idxs = rng.choice(len(imgs), size=sample_count, replace=False)

    cols = 3
    fig, axes = plt.subplots(sample_count, cols, figsize=(6 * cols, sample_count * 5))
    if sample_count == 1:
        axes = np.array([axes])

    img_h, img_w = imgs.shape[2], imgs.shape[3]

    for row_idx, sample_idx in enumerate(idxs):
        pred_value = float(preds[sample_idx])
        true_value = float(trues[sample_idx])
        rgb_img = np.transpose(imgs[sample_idx], (1, 2, 0)).astype(np.float32)
        rgb_img_clipped = np.clip(rgb_img, 0, 1)

        ax0 = axes[row_idx, 0]
        ax0.imshow(rgb_img_clipped)
        ax0.set_title(f"RGB\nTrue: {true_value:.3f} | Pred: {pred_value:.3f}")
        ax0.axis("off")

        ax1 = axes[row_idx, 1]
        ax1.imshow(masks[sample_idx, 0], cmap="gray")
        ax1.set_title("Phase Mask (GT)")
        ax1.axis("off")

        ax2 = axes[row_idx, 2]
        attn = attention_maps[sample_idx, 0]
        attn_resized = cv2.resize(attn, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        attn_min, attn_max = attn_resized.min(), attn_resized.max()
        if attn_max - attn_min > 1e-6:
            attn_norm = (attn_resized - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = attn_resized

        cmap_func = plt.get_cmap(heatmap_cmap)
        attn_colored = cmap_func(attn_norm)[:, :, :3].astype(np.float32)
        overlay = (1 - attention_alpha) * rgb_img_clipped + attention_alpha * attn_colored
        overlay = np.clip(overlay, 0, 1)

        ax2.imshow(overlay)
        ax2.set_title(f"Predicted Value: {pred_value:.3f}")
        ax2.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def prepare_run_dir(run_dir: Optional[str], mode: str) -> Path:
    if run_dir:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if mode == "test-only":
        candidates = sorted(Path("runs").glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No run directory found under runs/run_*.")
        return candidates[-1]

    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"runs/run_{dt}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_train_test(
    cfg: dict[str, Any],
    run_dir: Path,
    random_state: int,
    output_dir: Optional[str] = None,
    wandb_project: str = "im2prop",
    wandb_group: Optional[str] = None,
    run_name: Optional[str] = None,
    dummy_run: bool = False,
    dummy_samples: int = 32,
) -> None:
    run = wandb.init(project=wandb_project, group=wandb_group, name=run_name, config=cfg)
    cfg = dict(wandb.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(random_state)

    print_section("IM2PROP v2 - Runtime Configuration")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Random split seed: {random_state}")
    print(f"Dummy run: {dummy_run}")
    print(f"Patching: {'ENABLED' if cfg['ENABLE_PATCHING'] else 'DISABLED'}")
    if int(cfg.get("CV_FOLDS", 0)) > 1:
        print(f"CV: fold {int(cfg['CV_FOLD_IDX'])}/{int(cfg['CV_FOLDS'])}")

    full_df = pd.read_csv(cfg["CSV_V2_PATH"]).reset_index(drop=True)
    source_indices = list(range(len(full_df)))
    df = full_df
    if dummy_run:
        sampled = full_df.sample(n=min(len(full_df), max(3, int(dummy_samples))), random_state=random_state)
        source_indices = [int(i) for i in sampled.index.tolist()]
        df = sampled.reset_index(drop=True)

    split_path = run_dir / "split_meta.json"
    if split_path.exists():
        split_meta = load_json(split_path)
    else:
        if int(cfg.get("CV_FOLDS", 0)) > 1:
            split_meta = create_kfold_split_indices(df, int(cfg["CV_FOLDS"]), int(cfg["CV_FOLD_IDX"]), random_state)
        else:
            split_meta = create_split_indices(df, random_state)
        split_meta["source_indices"] = source_indices
        save_json(split_path, split_meta)

    train_df, val_df, test_df = select_splits_from_indices(df, split_meta)

    config_path = run_dir / "config.json"
    save_json(config_path, cfg)

    print_section("Dataset Summary")
    print(f"CSV path: {cfg['CSV_V2_PATH']}")
    print(f"Total rows used: {len(df)}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")

    to_tensor = transforms.Compose([transforms.ToTensor()])

    train_dataset = IM2PROPDataset_V2(train_df, cfg["IMAGE_DIR"], cfg["MASK_DIR"], transform_rgb=to_tensor, config=cfg)
    val_dataset = IM2PROPDataset_V2(val_df, cfg["IMAGE_DIR"], cfg["MASK_DIR"], transform_rgb=to_tensor, config=cfg)
    test_dataset = IM2PROPDataset_V2(test_df, cfg["IMAGE_DIR"], cfg["MASK_DIR"], transform_rgb=to_tensor, config=cfg)

    train_loader = DataLoader(train_dataset, batch_size=int(cfg["TRAIN_BATCH"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["VAL_BATCH"]), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg["TEST_BATCH"]), shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Test samples : {len(test_dataset)}")
    print(f"Auto-derived input size: {train_dataset.input_size}x{train_dataset.input_size}")

    model = build_model(cfg, device)
    freeze_encoder_if_needed(model, bool(cfg["FREEZE_ENCODER"]))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print_section("IM2PROP v2 Model Summary")
    print(f"Encoder: {cfg['ENCODER_NAME']} ({cfg['WEIGHTS_SOURCE']})")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")

    criterion = build_criterion(str(cfg["CRITERION"]))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(cfg["LEARNING_RATE"]))

    best_model_path = run_dir / "best_regression_model_v2.pt"
    best_val_loss = float("inf")

    val_frequency = 1 if dummy_run else int(cfg["VAL_FREQUENCY"])
    num_epochs = 1 if dummy_run else int(cfg["NUM_EPOCHS"])
    if val_frequency <= 0:
        raise ValueError("VAL_FREQUENCY must be >= 1")

    print_section("Training Start")
    print(f"Epochs: {num_epochs} | Validation every {val_frequency} epoch(s)")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        train_loss = train_one_epoch_v2(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        log_data: dict[str, Any] = {"epoch": epoch + 1, "train/loss": train_loss}

        if (epoch + 1) % val_frequency == 0:
            val_loss, val_mae, val_mse, val_rmse = validate_one_epoch_v2(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
            log_data.update({"val/loss": val_loss, "val/mae": val_mae, "val/mse": val_mse, "val/rmse": val_rmse})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print("Saved new best model checkpoint.")

        wandb.log(log_data)

    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    results = evaluate_test_set(model, test_loader, device)
    print_section("Test Results")
    print(f"MSE  : {results['mse']:.4f}")
    print(f"MAE  : {results['mae']:.4f}")
    print(f"RMSE : {results['rmse']:.4f}")

    if output_dir:
        vis_dir = Path(output_dir)
    else:
        vis_dir = run_dir

    vis_path = vis_dir / "test_predictions_phase_attention_pred_only.jpg"

    visualize_predictions(
        results=results,
        output_path=vis_path,
        num_samples=int(cfg["NUM_VIZ_SAMPLES"]),
        heatmap_cmap=str(cfg["HEATMAP_CMAP"]),
        attention_alpha=float(cfg["ATTENTION_ALPHA"]),
        random_state=random_state,
    )

    # Upload the saved test visualization so it is visible in the W&B run.
    wandb.log({"test/visualization": wandb.Image(str(vis_path))})
    wandb.save(str(vis_path))

    metrics_path = run_dir / "test_metrics.json"
    save_json(metrics_path, {"mse": results["mse"], "mae": results["mae"], "rmse": results["rmse"], "checkpoint": str(best_model_path)})

    # test_loader uses shuffle=False, so results["preds"]/["trues"] are already in split_meta["test_indices"] order.
    # Patching emits NUM_PATCHES rows per specimen; that misaligns the pairing
    # and would silently corrupt pooled CV metrics, so refuse rather than guess.
    if len(results["preds"]) != len(split_meta["test_indices"]):
        raise RuntimeError(
            f"{len(results['preds'])} predictions for {len(split_meta['test_indices'])} "
            "test specimens; set ENABLE_PATCHING=false for cross-validation runs"
        )
    predictions_path = run_dir / "test_predictions.json"
    save_json(
        predictions_path,
        {
            "test_indices": split_meta["test_indices"],
            "preds": [float(v) for v in results["preds"]],
            "trues": [float(v) for v in results["trues"]],
        },
    )

    wandb.log({"test/mse": results["mse"], "test/mae": results["mae"], "test/rmse": results["rmse"]})
    run.summary["best_model_path"] = str(best_model_path)
    run.summary["test_metrics_path"] = str(metrics_path)
    run.summary["visualization_path"] = str(vis_path)

    print_section("Output Summary")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Visualization: {vis_path}")
    print(f"Test metrics JSON: {metrics_path}")
    print(f"Test predictions JSON: {predictions_path}")

    wandb.finish()


def run_test_only(
    cfg: dict[str, Any],
    run_dir: Path,
    random_state: int,
    output_dir: Optional[str] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_section("IM2PROP v2 - Test-Only Configuration")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")

    split_path = run_dir / "split_meta.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split_meta.json not found in run dir: {run_dir}")
    split_meta = load_json(split_path)

    full_df = pd.read_csv(cfg["CSV_V2_PATH"]).reset_index(drop=True)
    source_indices = split_meta.get("source_indices")
    if source_indices is not None:
        df = full_df.iloc[source_indices].reset_index(drop=True)
    else:
        df = full_df
    _train_df, _val_df, test_df = select_splits_from_indices(df, split_meta)

    to_tensor = transforms.Compose([transforms.ToTensor()])
    test_dataset = IM2PROPDataset_V2(test_df, cfg["IMAGE_DIR"], cfg["MASK_DIR"], transform_rgb=to_tensor, config=cfg)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg["TEST_BATCH"]), shuffle=False)

    model = build_model(cfg, device)

    checkpoint_path = run_dir / "best_regression_model_v2.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results = evaluate_test_set(model, test_loader, device)

    print_section("Test Results")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"MSE  : {results['mse']:.4f}")
    print(f"MAE  : {results['mae']:.4f}")
    print(f"RMSE : {results['rmse']:.4f}")

    if output_dir:
        vis_dir = Path(output_dir)
    else:
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = Path(f"runs/test_{dt}")

    vis_path = vis_dir / "test_predictions_phase_attention_pred_only.jpg"

    visualize_predictions(
        results=results,
        output_path=vis_path,
        num_samples=int(cfg["NUM_VIZ_SAMPLES"]),
        heatmap_cmap=str(cfg["HEATMAP_CMAP"]),
        attention_alpha=float(cfg["ATTENTION_ALPHA"]),
        random_state=random_state,
    )

    print_section("Output Summary")
    print(f"Visualization: {vis_path}")
