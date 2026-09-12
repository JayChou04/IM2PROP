"""Default configuration for the IM2PROP pipeline.

This module extracts top-level globals and hyperparameters from the
original notebook-based workflow into a centralized Python dictionary.
"""

DEFAULT_CONFIG = {
    # File paths
    "IMAGE_DIR": "IM2PROP_data/HO_image/",
    "MASK_DIR": "IM2PROP_data/HO_mask/",
    "CSV_PATH": "IM2PROP_data/IM2PROP_dataset_homo_v1.csv",
    "CSV_V2_PATH": "IM2PROP_data/IM2PROP_dataset_homo_ratio_v1.csv",
    # CV pipeline parameters
    "MEDIAN_K": 5,
    "BG_SIGMA": 80,
    "CLAHE_CLIP": 3.0,
    "CLAHE_GRID": (8, 8),
    "GAUSS_K": (5, 5),
    "GAUSS_SIG": 1,
    "OPEN_K": 3,
    "OPEN_IT": 2,
    "CLOSE_K": 5,
    "CLOSE_IT": 2,
    "MIN_AREA_PIXELS": 200,
    # Dataset parameters
    "ENABLE_PATCHING": True,
    "NUM_PATCHES": 4,
    "TRAIN_BATCH": 8,
    "VAL_BATCH": 4,
    "TEST_BATCH": 4,
    "CV_FOLDS": 0,      # 0 = single holdout split; >1 = K-fold
    "CV_FOLD_IDX": 0,   # which fold is the test set, 0-based
    # Training parameters
    "NUM_EPOCHS": 30,
    "VAL_FREQUENCY": 2,
    "LEARNING_RATE": 1e-3,
    "CRITERION": "SmoothL1Loss",
    # Model parameters
    "ENCODER_NAME": "resnet50",
    "WEIGHTS_SOURCE": "micronet",
    "FREEZE_ENCODER": True,
    "USE_PHASE_RATIOS": True,
    "USE_PHASE_ATTENTION": True,
    "USE_PHASE_FEAT": True,
    # Visualization parameters
    "NUM_VIZ_SAMPLES": 6,
    "VIZ_COLS": 3,
    "ATTENTION_ALPHA": 0.3,
    "HEATMAP_ALPHA": 0.6,
    "HEATMAP_CMAP": "jet",
    # Sliding window inference parameters
    "SLIDING_WINDOW_STRIDE": 32,
    "SLIDING_WINDOW_MODE": "overlapping",
    "INFERENCE_IMAGE_PATH": "dss_2205/10X.png",
    # Runtime-derived globals from notebook (set during data loading)
    "DERIVED_INPUT_SIZE": None,
    "SLIDING_WINDOW_CROP_SIZE": None,
    "SLIDING_WINDOW_INPUT_SIZE": None,
    # Inference outputs
    "MAPPING_INFERENCE_DIR": "dss_2205/mapping_inference/",
}
