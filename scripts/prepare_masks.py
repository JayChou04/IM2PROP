"""Standalone entry point for the phase-mask extraction stage of the pipeline."""

import sys

from im2prop.config import DEFAULT_CONFIG
from im2prop.data.preprocess import process_dataset_phase_masks


def main() -> int:
    image_dir = DEFAULT_CONFIG["IMAGE_DIR"]
    mask_dir = DEFAULT_CONFIG["MASK_DIR"]
    csv_path = DEFAULT_CONFIG["CSV_PATH"]
    csv_v2_path = DEFAULT_CONFIG["CSV_V2_PATH"]

    print(f"IMAGE_DIR: {image_dir}")
    print(f"MASK_DIR: {mask_dir}")
    print(f"CSV_PATH: {csv_path}")
    print(f"CSV_V2_PATH: {csv_v2_path}")

    # use_existing_masks=True still fills in masks missing on a fresh clone.
    df, failed_images = process_dataset_phase_masks(
        image_dir=image_dir,
        mask_dir=mask_dir,
        csv_path=csv_path,
        csv_v2_path=csv_v2_path,
        use_existing_masks=True,
        config=DEFAULT_CONFIG,
    )

    print(f"Rows written: {len(df)}")
    print(f"Failed images: {failed_images}")

    return 1 if failed_images else 0


if __name__ == "__main__":
    sys.exit(main())
