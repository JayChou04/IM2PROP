#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root from this script's own location so it works from anywhere.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# No W&B account is needed to reproduce: runs log locally instead of syncing.
export WANDB_MODE=offline

SMOKE=false
DRY_RUN=false
REPEATS=3
prev=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=true ;;      # forwarded to stage 2: 1 combo x 1 seed x 1 fold x 30 epochs
    --dry-run) DRY_RUN=true ;;
  esac
  # Stage 3 must expect the same repeat count the sweep was run with, or a
  # deliberately shortened run would be reported as incomplete.
  [[ "$prev" == "--repeats" ]] && REPEATS="$arg"
  prev="$arg"
done

banner() {
  echo ""
  echo "==== $* ===="
}

banner "Stage 0: preflight checks"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. Install it: https://docs.astral.sh/uv/" >&2
  exit 1
fi
echo "uv found: $(command -v uv)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi present but query failed"
else
  echo "WARNING: nvidia-smi not found; CUDA availability could not be confirmed, training will fall back to CPU"
fi

if [[ "$DRY_RUN" != "true" ]]; then
  banner "Stage 1: phase mask extraction"
  uv run scripts/prepare_masks.py
fi

banner "Stage 2: paired ablation sweep"
bash scripts/run_phase_combinations.sh "$@"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete: only stages 0 and 2 ran."
  exit 0
fi

banner "Stage 3: sweep verification"
if [[ "$SMOKE" == "true" ]]; then
  echo "Skipping: --smoke runs a reduced matrix, so the completeness gate is expected to fail."
else
  uv run scripts/sweep_status.py --report --repeats "${REPEATS}"
fi

banner "Stage 4: CV aggregation"
uv run scripts/aggregate_cv.py

banner "Stage 5: figure generation"
# Both stages write straight into docs/assets/, which GitHub Pages serves.
uv run analysis/mae_visualization/visualize.py

echo ""
echo "Full run: ~15 hours for 360 runs on a single CUDA GPU. The sweep is resume-safe, so an interrupted job can be restarted with the same command."
