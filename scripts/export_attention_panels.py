"""Export phase-attention panels for the project page.

Reads one finished sweep run without modifying it and writes images to
docs/static/images/attention/. Nothing goes under runs/: a duplicated run there
would make aggregate_cv.py stop.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from im2prop.data.dataset import IM2PROPDataset_V2
from im2prop.training.pipeline import (
    build_model,
    evaluate_test_set,
    load_json,
    save_json,
    select_splits_from_indices,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "docs" / "static" / "images" / "attention"

# Combo 4 (attention + phase features, no ratios) at 80 epochs is the lowest-MAE
# cell of the results table. 6311 is the first repeat seed drawn from META_SEED=0.
FLAGS = {"USE_PHASE_RATIOS": False, "USE_PHASE_ATTENTION": True, "USE_PHASE_FEAT": True}
EPOCHS = 80
SEED = 6311
FOLD = 0


def find_run() -> Path:
    """The single finished run matching FLAGS, EPOCHS, SEED and FOLD."""
    hits = []
    for run_dir in sorted(p for p in RUNS_DIR.glob("*") if p.is_dir()):
        try:
            cfg = load_json(run_dir / "config.json")
            split = load_json(run_dir / "split_meta.json")
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if (all(bool(cfg[k]) == v for k, v in FLAGS.items())
                and int(cfg["NUM_EPOCHS"]) == EPOCHS
                and int(split["random_state"]) == SEED
                and int(split.get("cv_fold_idx", 0)) == FOLD
                and (run_dir / "best_regression_model_v2.pt").exists()
                and (run_dir / "test_predictions.json").exists()):
            hits.append(run_dir)
    if len(hits) != 1:
        sys.exit(f"Expected one matching run under {RUNS_DIR}, found {[h.name for h in hits]}")
    return hits[0]


def pick_specimens(trues: np.ndarray) -> list[int]:
    """Positions of the lowest, median and highest measured hardness."""
    order = np.argsort(trues)
    return [int(order[0]), int(order[len(order) // 2]), int(order[-1])]


def attention_overlay(rgb: np.ndarray, attn: np.ndarray, cmap: str, alpha: float) -> np.ndarray:
    """Upsampled, min-max-normalised attention blended over the micrograph, as in visualize_predictions."""
    h, w = rgb.shape[:2]
    attn = cv2.resize(attn, (w, h), interpolation=cv2.INTER_CUBIC)
    span = attn.max() - attn.min()
    if span > 1e-6:
        attn = (attn - attn.min()) / span
    colored = plt.get_cmap(cmap)(attn)[:, :, :3].astype(np.float32)
    return np.clip((1 - alpha) * rgb + alpha * colored, 0, 1)


def main() -> int:
    run_dir = find_run()
    cfg = load_json(run_dir / "config.json")
    split = load_json(run_dir / "split_meta.json")
    saved = load_json(run_dir / "test_predictions.json")
    print(f"Run: {run_dir.name}")

    df = pd.read_csv(ROOT / cfg["CSV_V2_PATH"]).reset_index(drop=True)
    if split.get("source_indices") is not None:
        df = df.iloc[split["source_indices"]].reset_index(drop=True)
    _, _, test_df = select_splits_from_indices(df, split)

    dataset = IM2PROPDataset_V2(test_df, ROOT / cfg["IMAGE_DIR"], ROOT / cfg["MASK_DIR"],
                                transform_rgb=transforms.ToTensor(), config=cfg)
    loader = DataLoader(dataset, batch_size=int(cfg["TEST_BATCH"]), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(run_dir / "best_regression_model_v2.pt", map_location=device))
    model.eval()
    results = evaluate_test_set(model, loader, device)

    # Same checkpoint on the same specimens must reproduce what the sweep saved,
    # which confirms the attention maps come from the model in the results table.
    drift = float(np.max(np.abs(results["preds"] - np.asarray(saved["preds"]))))
    if drift > 1e-3:
        sys.exit(f"Predictions differ from {run_dir.name}/test_predictions.json by {drift:.4f} GPa")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specimens = []
    for pos in pick_specimens(results["trues"]):
        image_id = str(test_df.iloc[pos]["image_id"])
        rgb = np.clip(results["imgs"][pos].transpose(1, 2, 0), 0, 1).astype(np.float32)
        shutil.copy(ROOT / cfg["IMAGE_DIR"] / f"{image_id}.jpg", OUT_DIR / f"{image_id}_micrograph.jpg")
        attn = attention_overlay(rgb, results["attn"][pos, 0], str(cfg["HEATMAP_CMAP"]),
                                 float(cfg["ATTENTION_ALPHA"]))
        Image.fromarray((attn * 255).round().astype(np.uint8)).save(OUT_DIR / f"{image_id}_attention.jpg", quality=90)
        specimens.append({"image_id": image_id, "measured_gpa": round(float(results["trues"][pos]), 4)})
        print(f"  {image_id}: measured {specimens[-1]['measured_gpa']:.3f} GPa")

    save_json(OUT_DIR / "panels.json", {"run": run_dir.name, "combo": 4, "epochs": EPOCHS,
                                        "seed": SEED, "fold": FOLD, "specimens": specimens})
    print(f"Wrote {2 * len(specimens)} panels and panels.json to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
