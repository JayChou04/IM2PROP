"""Dataset utilities for IM2PROP.

This module contains the dual-branch dataset used in the notebook workflow,
refactored into a reusable package module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from im2prop.config import DEFAULT_CONFIG


def _get_cfg(config: Optional[dict[str, Any]], key: str, default: Any = None) -> Any:
    """Read a value from user config, then DEFAULT_CONFIG, then fallback default."""
    if config is not None and key in config:
        return config[key]
    if key in DEFAULT_CONFIG:
        return DEFAULT_CONFIG[key]
    return default


def _auto_detect_image_size(
    image_dir: str | Path,
    csv_data: pd.DataFrame,
    image_id_col: str = "image_id",
    image_suffix: str = ".jpg",
) -> tuple[int, int]:
    """Auto-detect image dimensions from the first available image in csv_data."""
    image_dir = Path(image_dir)

    for _, row in csv_data.iterrows():
        img_path = image_dir / f"{row[image_id_col]}{image_suffix}"
        if img_path.exists():
            with Image.open(img_path) as img:
                width, height = img.size
            return width, height

    raise FileNotFoundError(f"No images found in {image_dir}")


def _compute_patch_boxes(
    img_w: int,
    img_h: int,
    num_patches: int,
) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Compute patch crop boxes for a square grid of num_patches."""
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"num_patches must be a perfect square (1, 4, 9, 16, ...), got {num_patches}"
        )

    patch_w = img_w // grid_size
    patch_h = img_h // grid_size

    if img_w % grid_size != 0 or img_h % grid_size != 0:
        raise ValueError(
            f"Image size ({img_w}x{img_h}) is not evenly divisible by grid_size ({grid_size}). "
            f"Remainder: width={img_w % grid_size}, height={img_h % grid_size}"
        )

    boxes = [
        (x * patch_w, y * patch_h, (x + 1) * patch_w, (y + 1) * patch_h)
        for y in range(grid_size)
        for x in range(grid_size)
    ]
    return boxes, patch_w, patch_h


class IM2PROPDataset_V2(Dataset):
    """Dual-branch dataset: RGB image + phase mask + phase ratios.

    Modes:
    - Patching ON: tiles each image into num_patches patches.
    - Patching OFF: uses full image as-is.
    """

    def __init__(
        self,
        csv_data: pd.DataFrame,
        image_dir: str | Path,
        mask_dir: str | Path,
        enable_patching: Optional[bool] = None,
        num_patches: Optional[int] = None,
        transform_rgb: Any = None,
        transform_mask: Any = None,
        return_info: bool = False,
        config: Optional[dict[str, Any]] = None,
        image_id_col: str = "image_id",
        target_col: str = "hardness_value",
        light_ratio_col: str = "light_phase_ratio",
        dark_ratio_col: str = "dark_phase_ratio",
        image_suffix: str = ".jpg",
        mask_suffix: str = "_mask.jpg",
    ) -> None:
        self.data = csv_data.reset_index(drop=True)
        self.img_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.enable_patching = (
            _get_cfg(config, "ENABLE_PATCHING") if enable_patching is None else enable_patching
        )
        self.num_patches = (
            int(_get_cfg(config, "NUM_PATCHES")) if num_patches is None else int(num_patches)
        )

        self.transform_rgb = transform_rgb
        self.transform_mask = transform_mask
        self.return_info = return_info

        self.image_id_col = image_id_col
        self.target_col = target_col
        self.light_ratio_col = light_ratio_col
        self.dark_ratio_col = dark_ratio_col
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        self.img_w, self.img_h = _auto_detect_image_size(
            image_dir=self.img_dir,
            csv_data=self.data,
            image_id_col=self.image_id_col,
            image_suffix=self.image_suffix,
        )

        if self.enable_patching:
            self.boxes, self.patch_w, self.patch_h = _compute_patch_boxes(
                self.img_w,
                self.img_h,
                self.num_patches,
            )
            self.input_size = self.patch_w
        else:
            self.boxes = None
            self.input_size = self.img_w

    def __len__(self) -> int:
        if self.enable_patching:
            return len(self.data) * self.num_patches
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.enable_patching:
            base_idx = idx // self.num_patches
            patch_id = idx % self.num_patches
        else:
            base_idx = idx
            patch_id = 0

        row = self.data.iloc[base_idx]
        img_name = row[self.image_id_col]

        target = row[self.target_col]
        light_ratio = row[self.light_ratio_col]
        dark_ratio = row[self.dark_ratio_col]

        img_path = self.img_dir / f"{img_name}{self.image_suffix}"
        mask_path = self.mask_dir / f"{img_name}{self.mask_suffix}"

        rgb_image = Image.open(img_path).convert("RGB")
        phase_mask = Image.open(mask_path).convert("L")

        if self.enable_patching:
            box = self.boxes[patch_id]
            rgb_tile = rgb_image.crop(box)
            mask_tile = phase_mask.crop(box)
        else:
            rgb_tile = rgb_image
            mask_tile = phase_mask

        if self.transform_rgb:
            rgb_tile = self.transform_rgb(rgb_tile)
        else:
            rgb_tile = transforms.ToTensor()(rgb_tile)

        if self.transform_mask:
            mask_tile = self.transform_mask(mask_tile)
        else:
            mask_tile = torch.from_numpy(np.array(mask_tile, dtype=np.float32)).unsqueeze(0)

        mask_tile = mask_tile / 255.0
        phase_ratios = torch.tensor(
            [light_ratio / 100.0, dark_ratio / 100.0],
            dtype=torch.float32,
        )
        target = torch.tensor(target, dtype=torch.float32)

        if self.return_info:
            return rgb_tile, mask_tile, phase_ratios, target, (img_name, patch_id)
        return rgb_tile, mask_tile, phase_ratios, target
