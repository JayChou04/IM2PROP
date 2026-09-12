"""Preprocessing utilities for IM2PROP.

This module contains standalone utilities for directory creation, CSV loading,
and OpenCV-based phase mask extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from im2prop.config import DEFAULT_CONFIG


def _get_cfg(config: Optional[dict[str, Any]], key: str, default: Any = None) -> Any:
    """Read a value from user config, then DEFAULT_CONFIG, then fallback default."""
    if config is not None and key in config:
        return config[key]
    if key in DEFAULT_CONFIG:
        return DEFAULT_CONFIG[key]
    return default


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it as Path."""
    path_obj = Path(path)
    path_obj.mkdir(exist_ok=True, parents=True)
    return path_obj


def read_dataset_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a dataset CSV file."""
    return pd.read_csv(csv_path)


def extract_phase_mask(
    img_path: str | Path,
    *,
    median_k: Optional[int] = None,
    bg_sigma: Optional[float] = None,
    clahe_clip: Optional[float] = None,
    clahe_grid: Optional[tuple[int, int]] = None,
    gauss_k: Optional[tuple[int, int]] = None,
    gauss_sig: Optional[float] = None,
    open_k: Optional[int] = None,
    open_it: Optional[int] = None,
    close_k: Optional[int] = None,
    close_it: Optional[int] = None,
    min_area_pixels: Optional[int] = None,
    config: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, float, float]:
    """Apply CV pipeline to extract a binary phase mask and phase ratios.

    Returns:
    - mask_filtered: uint8 mask [H, W], where 255=light phase, 0=dark phase
    - light_ratio: percentage of light phase
    - dark_ratio: percentage of dark phase
    """
    median_k = int(_get_cfg(config, "MEDIAN_K") if median_k is None else median_k)
    bg_sigma = float(_get_cfg(config, "BG_SIGMA") if bg_sigma is None else bg_sigma)
    clahe_clip = float(_get_cfg(config, "CLAHE_CLIP") if clahe_clip is None else clahe_clip)
    clahe_grid = tuple(_get_cfg(config, "CLAHE_GRID") if clahe_grid is None else clahe_grid)
    gauss_k = tuple(_get_cfg(config, "GAUSS_K") if gauss_k is None else gauss_k)
    gauss_sig = float(_get_cfg(config, "GAUSS_SIG") if gauss_sig is None else gauss_sig)
    open_k = int(_get_cfg(config, "OPEN_K") if open_k is None else open_k)
    open_it = int(_get_cfg(config, "OPEN_IT") if open_it is None else open_it)
    close_k = int(_get_cfg(config, "CLOSE_K") if close_k is None else close_k)
    close_it = int(_get_cfg(config, "CLOSE_IT") if close_it is None else close_it)
    min_area_pixels = int(
        _get_cfg(config, "MIN_AREA_PIXELS") if min_area_pixels is None else min_area_pixels
    )

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_med = cv2.medianBlur(img_gray, median_k)

    bg = cv2.GaussianBlur(img_med, (0, 0), bg_sigma).astype(np.float32)
    bg = np.clip(bg, 1, None)
    img_norm = (img_med.astype(np.float32) / bg * 128).clip(0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    img_clahe = clahe.apply(img_norm)

    img_smooth = cv2.GaussianBlur(img_clahe, gauss_k, gauss_sig)

    _, mask_otsu = cv2.threshold(
        img_smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    mask_opened = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, open_kernel, iterations=open_it)
    mask_clean = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, close_kernel, iterations=close_it)

    dark_phase_mask = (mask_clean == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark_phase_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_filtered = mask_clean.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_pixels:
            cv2.drawContours(mask_filtered, [contour], 0, 255, -1)

    light_pixels = np.sum(mask_filtered == 255)
    dark_pixels = np.sum(mask_filtered == 0)
    total_pixels = mask_filtered.size

    light_ratio = (light_pixels / total_pixels) * 100.0
    dark_ratio = (dark_pixels / total_pixels) * 100.0

    return mask_filtered, light_ratio, dark_ratio


def process_dataset_phase_masks(
    *,
    image_dir: str | Path,
    mask_dir: str | Path,
    csv_path: str | Path,
    csv_v2_path: str | Path,
    image_id_col: str = "image_id",
    light_ratio_col: str = "light_phase_ratio",
    dark_ratio_col: str = "dark_phase_ratio",
    image_suffix: str = ".jpg",
    mask_suffix: str = "_mask.jpg",
    use_existing_masks: bool = True,
    config: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run phase-mask extraction for all rows and save an augmented CSV.

    Returns:
    - updated dataframe
    - list of failed image IDs

    When use_existing_masks=True, an existing mask file is reused if present.
    When use_existing_masks=False, masks are always regenerated from source images.
    """
    image_dir = Path(image_dir)
    mask_dir = ensure_directory(mask_dir)

    df = read_dataset_csv(csv_path)

    light_ratios: list[float] = []
    dark_ratios: list[float] = []
    failed_images: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting phase masks"):
        img_id = row[image_id_col]
        img_path = image_dir / f"{img_id}{image_suffix}"
        mask_path = mask_dir / f"{img_id}{mask_suffix}"

        try:
            if use_existing_masks and mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Cannot read existing mask: {mask_path}")
                # Normalize potentially compressed masks to strict binary values.
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                light_ratio = float(np.mean(mask == 255) * 100.0)
                dark_ratio = float(np.mean(mask == 0) * 100.0)
            else:
                mask, light_ratio, dark_ratio = extract_phase_mask(img_path, config=config)
                cv2.imwrite(str(mask_path), mask)
            light_ratios.append(light_ratio)
            dark_ratios.append(dark_ratio)
        except Exception:
            failed_images.append(str(img_id))
            light_ratios.append(np.nan)
            dark_ratios.append(np.nan)

    df[light_ratio_col] = light_ratios
    df[dark_ratio_col] = dark_ratios

    df.to_csv(csv_v2_path, index=False)
    return df, failed_images
