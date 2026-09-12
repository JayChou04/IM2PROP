"""Pool K-fold predictions into per-configuration cross-validated MAE.

Each run covers one fold, so its own test_metrics.json describes ~19 specimens.
Pooling the folds of one (combo, epoch, seed) reconstructs a prediction for all
94 specimens, which is the number worth reporting.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
# Written straight into docs/ because GitHub Pages serves only that subtree;
# a second copy elsewhere would just be a source of drift.
OUT_DIR = ROOT / "docs" / "assets"

COMBOS = [(r, a, f) for r in (0, 1) for a in (0, 1) for f in (0, 1)]
COMBO_NAMES = [
    "Combo 1 Baseline", "Combo 2 +Phase feat", "Combo 3 +Attention",
    "Combo 4 +Attn+feat", "Combo 5 +Ratios", "Combo 6 +Ratio+feat",
    "Combo 7 +Ratio+attn", "Combo 8 Full IM2PROP",
]


def load_folds() -> dict[tuple, dict[int, float]]:
    """Map (combo, epoch, seed) -> {specimen index: signed residual pred-true}."""
    acc: dict[tuple, dict[int, float]] = defaultdict(dict)
    for run_dir in sorted(p for p in RUNS_DIR.glob("*") if p.is_dir()):
        try:
            cfg = json.loads((run_dir / "config.json").read_text())
            pred = json.loads((run_dir / "test_predictions.json").read_text())
            split = json.loads((run_dir / "split_meta.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        key = (int(cfg["USE_PHASE_RATIOS"]), int(cfg["USE_PHASE_ATTENTION"]),
               int(cfg["USE_PHASE_FEAT"]))
        ident = (COMBOS.index(key) + 1, int(cfg["NUM_EPOCHS"]), int(split["random_state"]))
        for idx, p, t in zip(pred["test_indices"], pred["preds"], pred["trues"]):
            # A specimen must appear once per (combo, epoch, seed); a repeat means
            # duplicated folds, which would silently bias the pooled mean.
            if int(idx) in acc[ident]:
                sys.exit(f"Duplicate specimen {idx} in {ident} — check for duplicated folds.")
            acc[ident][int(idx)] = float(p) - float(t)
    return acc


def main() -> int:
    acc = load_folds()
    if not acc:
        sys.exit(f"No pooled runs found under {RUNS_DIR}. Run the sweep first.")

    n_specimens = max(len(v) for v in acc.values())

    # Absolute error per specimen, one vector per repeat, keyed by (combo, epoch).
    per_cell: dict[tuple, list[np.ndarray]] = defaultdict(list)
    rows = []
    for (combo, ep, seed), resid in sorted(acc.items()):
        if len(resid) != n_specimens:
            print(f"  WARNING: combo{combo} {ep}ep seed{seed} covers {len(resid)}/"
                  f"{n_specimens} specimens — incomplete fold set, excluded")
            continue
        r = np.array([resid[i] for i in sorted(resid)])
        per_cell[(combo, ep)].append(np.abs(r))
        rows.append(dict(combo=combo, name=COMBO_NAMES[combo - 1], epochs=ep, seed=seed,
                         n=len(r), cv_mae=float(np.abs(r).mean()),
                         cv_rmse=float(np.sqrt((r ** 2).mean()))))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "cv_per_repeat.csv", index=False)
    print(f"Pooled {len(rows)} (combo, epoch, repeat) estimates over "
          f"{n_specimens} specimens each.\n")

    # A cell's CV MAE: per-specimen error averaged over the repeats, then over specimens.
    summary = [dict(combo=combo, name=COMBO_NAMES[combo - 1], epochs=ep, n_repeats=len(vecs),
                    cv_mae=float(np.mean(vecs, axis=0).mean()))
               for (combo, ep), vecs in sorted(per_cell.items())]
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT_DIR / "cv_summary.csv", index=False)
    print("=== Cross-validated MAE (GPa) ===")
    print(sm.pivot(index="name", columns="epochs", values="cv_mae").round(4).to_string())
    print(f"\nWrote cv_per_repeat.csv and cv_summary.csv to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
