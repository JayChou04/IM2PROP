"""Scan runs/ for completed phase-ablation sweep runs and check or report their status."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

COMBOS = [
    (ratios, attention, feat)
    for ratios in (False, True)
    for attention in (False, True)
    for feat in (False, True)
]
EPOCH_LIST = (30, 50, 80)
CV_FOLDS = 5
DEFAULT_REPEATS = 3


def str2bool(value: str) -> bool:
    """Accept 0/1 or true/false (case-insensitive)."""
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def run_key(run_dir: Path) -> tuple[bool, bool, bool, int, int, int] | None:
    """Return the ablation key for a run dir, or None if it's not complete."""
    config_path = run_dir / "config.json"
    split_path = run_dir / "split_meta.json"
    metrics_path = run_dir / "test_metrics.json"
    if not (config_path.exists() and split_path.exists() and metrics_path.exists()):
        return None
    try:
        config = json.loads(config_path.read_text())
        split_meta = json.loads(split_path.read_text())
        return (
            bool(config["USE_PHASE_RATIOS"]),
            bool(config["USE_PHASE_ATTENTION"]),
            bool(config["USE_PHASE_FEAT"]),
            int(config["NUM_EPOCHS"]),
            int(split_meta["random_state"]),
            int(split_meta.get("cv_fold_idx", 0)),  # absent on pre-CV holdout runs
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return None  # malformed run dir; don't let one bad run crash the scan


def find_complete_runs() -> list[tuple[bool, bool, bool, int, int, int]]:
    """Collect the key tuple of every complete run under runs/."""
    if not RUNS_DIR.is_dir():
        return []
    keys = []
    for run_dir in sorted(p for p in RUNS_DIR.glob("*") if p.is_dir()):
        key = run_key(run_dir)
        if key is not None:
            keys.append(key)
    return keys


def combo_label(ratios: bool, attention: bool, feat: bool) -> str:
    """1-based combo id (paper numbering) plus the RAF booleans, e.g. 'combo05 r1a0f0'."""
    combo_id = COMBOS.index((ratios, attention, feat)) + 1
    return f"combo{combo_id:02d} r{int(ratios)}a{int(attention)}f{int(feat)}"


def cmd_check(values: list[str]) -> int:
    ratios, attention, feat, epochs, seed, fold = values
    target = (
        str2bool(ratios),
        str2bool(attention),
        str2bool(feat),
        int(epochs),
        int(seed),
        int(fold),
    )
    if target in find_complete_runs():
        return 0
    print(
        f"No completed run for {combo_label(*target[:3])} @ {target[3]} epochs, "
        f"seed {target[4]}, fold {target[5]}"
    )
    return 1


def cmd_report(repeats: int) -> int:
    EXPECTED_PER_CELL = CV_FOLDS * repeats
    keys = find_complete_runs()
    cell_counts = Counter((r, a, f, ep) for (r, a, f, ep, _seed, _fold) in keys)

    header = f"{'combo':<20}" + "".join(f"{ep:>8}ep" for ep in EPOCH_LIST)
    print(header)
    bad_cells = []
    for ratios, attention, feat in COMBOS:
        row = f"{combo_label(ratios, attention, feat):<20}"
        for ep in EPOCH_LIST:
            count = cell_counts.get((ratios, attention, feat, ep), 0)
            row += f"{count:>10}"
            if count != EXPECTED_PER_CELL:
                bad_cells.append((ratios, attention, feat, ep, count))
        print(row)

    total_cells = len(COMBOS) * len(EPOCH_LIST)
    ok_cells = total_cells - len(bad_cells)
    print(
        f"\nSummary: {len(keys)}/{total_cells * EXPECTED_PER_CELL} completed runs, "
        f"{ok_cells}/{total_cells} cells at expected count ({EXPECTED_PER_CELL})."
    )

    if bad_cells:
        print("Cells not at expected count:")
        for ratios, attention, feat, ep, count in bad_cells:
            status = "missing" if count < EXPECTED_PER_CELL else "over-count"
            print(f"  {combo_label(ratios, attention, feat)} @ {ep} epochs: found {count}, need {EXPECTED_PER_CELL} ({status})")

    # A cell can total EXPECTED_PER_CELL with a duplicated fold and a missing one
    # (e.g. fold 0 run twice, fold 4 never run), which would silently corrupt the
    # pooled metric -- so also check each (combo, epoch, seed) has 5 distinct folds.
    fold_sets: dict[tuple[bool, bool, bool, int, int], set[int]] = {}
    for ratios, attention, feat, ep, seed, fold in keys:
        fold_sets.setdefault((ratios, attention, feat, ep, seed), set()).add(fold)

    bad_fold_groups = sorted(
        (ratios, attention, feat, ep, seed, tuple(sorted(folds)))
        for (ratios, attention, feat, ep, seed), folds in fold_sets.items()
        if len(folds) != CV_FOLDS
    )
    if bad_fold_groups:
        print("Groups with missing/duplicated folds:")
        for ratios, attention, feat, ep, seed, folds in bad_fold_groups:
            print(
                f"  {combo_label(ratios, attention, feat)} @ {ep} epochs, seed {seed}: "
                f"folds present {list(folds)}, need {CV_FOLDS} distinct"
            )

    return 0 if not bad_cells and not bad_fold_groups else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or report phase-ablation sweep run status")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        nargs=6,
        metavar=("RATIOS", "ATTENTION", "FEAT", "EPOCHS", "SEED", "FOLD"),
        help="exit 0 if a completed run with this exact key exists",
    )
    mode.add_argument("--report", action="store_true", help="print the combo x epoch completion matrix")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"repeat-seeds the sweep was run with (default {DEFAULT_REPEATS})")
    args = parser.parse_args()

    if args.report:
        return cmd_report(args.repeats)
    return cmd_check(args.check)


if __name__ == "__main__":
    sys.exit(main())
