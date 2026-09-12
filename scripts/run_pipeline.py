"""Unified IM2PROP pipeline CLI.

Usage:
- Train + test in one run (default):
  uv run scripts/run_pipeline.py --MODE train-test
- Train + test with overrides:
  uv run scripts/run_pipeline.py --MODE train-test --TRAIN_BATCH 16 --USE_PHASE_ATTENTION false
- Retest an existing run directory:
  uv run scripts/run_pipeline.py --MODE test-only --RUN_DIR runs/run_20260309_173631
"""

from __future__ import annotations

import argparse
import ast
import copy
import random
from pathlib import Path
from typing import Any

from im2prop.config import DEFAULT_CONFIG
from im2prop.data.preprocess import process_dataset_phase_masks
from im2prop.training.pipeline import (
    load_json,
    prepare_run_dir,
    run_test_only,
    run_train_test,
)


def str2bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y", "t"}:
        return True
    if value_lower in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_value(raw: str, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return str2bool(raw)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw)
    if isinstance(default_value, float):
        return float(raw)
    if isinstance(default_value, tuple):
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, (tuple, list)):
            raise ValueError(f"Expected tuple/list literal, got: {raw}")
        return tuple(parsed)
    if default_value is None:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw
    return raw


def build_parser(default_config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified IM2PROP train/test pipeline")

    for key in default_config:
        parser.add_argument(f"--{key}", type=str, default=None, help=f"Override config key {key}")

    parser.add_argument("--MODE", type=str, default="train-test", choices=["train-test", "test-only"])
    parser.add_argument("--RUN_DIR", type=str, default=None, help="Reuse an existing run directory.")
    parser.add_argument(
        "--RANDOM_STATE",
        type=int,
        default=None,
        help="Random split seed. If omitted, auto-sampled from 0..10000.",
    )
    parser.add_argument("--OUTPUT_DIR", type=str, default=None)
    parser.add_argument("--WANDB_PROJECT", type=str, default="im2prop")
    parser.add_argument("--WANDB_GROUP", type=str, default=None)
    parser.add_argument("--RUN_NAME", type=str, default=None)
    parser.add_argument("--DUMMY_RUN", type=str, default="false")
    parser.add_argument("--DUMMY_SAMPLES", type=int, default=32)
    parser.add_argument(
        "--USE_OLD_MASKS",
        type=str,
        default="false",
        help="If true, reuse existing mask files; if false, regenerate masks.",
    )

    return parser


def build_config_from_args(
    args: argparse.Namespace,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    for key, default_value in base_config.items():
        raw_value = getattr(args, key)
        if raw_value is not None:
            cfg[key] = parse_value(raw_value, default_value)
    return cfg


def main() -> None:
    parser = build_parser(DEFAULT_CONFIG)
    args = parser.parse_args()

    run_dir = prepare_run_dir(args.RUN_DIR, args.MODE)

    base_config = DEFAULT_CONFIG
    config_json_path = run_dir / "config.json"
    if args.MODE == "test-only" and config_json_path.exists():
        base_config = load_json(config_json_path)

    cfg = build_config_from_args(args, base_config)

    dummy_run = str2bool(args.DUMMY_RUN)
    use_old_masks = str2bool(args.USE_OLD_MASKS)
    random_state = args.RANDOM_STATE if args.RANDOM_STATE is not None else random.randint(0, 10000)
    print(f"Using RANDOM_STATE={random_state}", flush=True)

    print(f"Preparing masks (USE_OLD_MASKS={use_old_masks})...")
    _df_masks, failed_images = process_dataset_phase_masks(
        image_dir=cfg["IMAGE_DIR"],
        mask_dir=cfg["MASK_DIR"],
        csv_path=cfg["CSV_PATH"],
        csv_v2_path=cfg["CSV_V2_PATH"],
        config=cfg,
        use_existing_masks=use_old_masks,
    )
    print(f"Mask preprocessing finished. Failed images: {len(failed_images)}")

    if args.MODE == "train-test":
        run_train_test(
            cfg=cfg,
            run_dir=run_dir,
            random_state=random_state,
            output_dir=args.OUTPUT_DIR,
            wandb_project=args.WANDB_PROJECT,
            wandb_group=args.WANDB_GROUP,
            run_name=args.RUN_NAME,
            dummy_run=dummy_run,
            dummy_samples=args.DUMMY_SAMPLES,
        )
        return

    run_test_only(
        cfg=cfg,
        run_dir=Path(run_dir),
        random_state=random_state,
        output_dir=args.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
