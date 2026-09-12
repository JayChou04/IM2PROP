#!/usr/bin/env bash
set -euo pipefail

# 5-fold CV: 8 phase-ablation combos x {30,50,80} epochs x N repeat-seeds x 5 folds.
# With the default 3 repeats that is 360 runs, and each (combo, epoch) cell holds
# 15; aggregate_cv.py pools a cell into the reported metric. Same SEEDS reused
# across every combo and every epoch setting, so each configuration sees an
# identical fold partition and identical initial weights -- that's what makes it paired.
#
# Seeds are derived from META_SEED rather than hard-coded, so the list is
# reproducible from a single documented number. Change META_SEED to draw a
# different sample. Python's Mersenne Twister sequence is stable across 3.x.
META_SEED=0
NUM_REPEATS=3
CV_FOLDS=5
EPOCH_LIST=(30 50 80)

DRY_RUN=false
SMOKE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --smoke) SMOKE=true; shift ;;
    --repeats)
      NUM_REPEATS="${2:-}"
      [[ "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats needs a positive integer" >&2; exit 1; }
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--repeats N] [--smoke] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

# Seeds are drawn in a fixed order, so --repeats 1 is a prefix of --repeats 3:
# raising the count later reuses the earlier runs instead of discarding them.
mapfile -t SEEDS < <(uv run python -c "
import random
random.seed(${META_SEED})
for _ in range(${NUM_REPEATS}):
    print(random.randint(0, 10000))
")
echo "META_SEED=${META_SEED}, repeats=${NUM_REPEATS} -> SEEDS=(${SEEDS[*]})"

total=0
executed=0
skipped=0

for ep in "${EPOCH_LIST[@]}"; do
  combo_id=1
  for ratios in false true; do
    for attention in false true; do
      for feat in false true; do
        r01=$([[ "$ratios" == "true" ]] && echo 1 || echo 0)
        a01=$([[ "$attention" == "true" ]] && echo 1 || echo 0)
        f01=$([[ "$feat" == "true" ]] && echo 1 || echo 0)
        printf -v combo_tag "combo%02d_r%s_a%s_f%s" "$combo_id" "$r01" "$a01" "$f01"

        for seed in "${SEEDS[@]}"; do
          for ((fold = 0; fold < CV_FOLDS; fold++)); do
            if [[ "$SMOKE" == "true" ]]; then
              # First seed of the derived list, not a literal, so --smoke survives a META_SEED change.
              if [[ "$combo_id" -ne 1 || "$seed" -ne "${SEEDS[0]}" || "$ep" -ne 30 || "$fold" -ne 0 ]]; then
                continue
              fi
            fi

            total=$((total + 1))
            run_name="${combo_tag}_seed${seed}_fold${fold}_${ep}_epochs"
            wandb_group="${combo_tag}_fold${fold}_${ep}_epochs"

            if [[ "$DRY_RUN" == "true" ]]; then
              echo "[DRY RUN] ${run_name} (WANDB_GROUP=${wandb_group})"
              continue
            fi

            if uv run scripts/sweep_status.py --check "$r01" "$a01" "$f01" "$ep" "$seed" "$fold"; then
              echo "Skipping ${run_name}: already complete"
              skipped=$((skipped + 1))
              continue
            fi

            echo "Running ${run_name} (WANDB_GROUP=${wandb_group})"

            # Tee the console output: offline wandb never writes files/output.log,
            # and visualize.py needs these lines for the learning-curve figure.
            run_log="$(mktemp)"

            # No --RUN_DIR: run_pipeline.py must mint its own run_<timestamp> dir.
            uv run scripts/run_pipeline.py \
              --MODE train-test --USE_OLD_MASKS true \
              --ENABLE_PATCHING false --NUM_EPOCHS "${ep}" --RANDOM_STATE "${seed}" \
              --USE_PHASE_RATIOS "${ratios}" --USE_PHASE_ATTENTION "${attention}" \
              --USE_PHASE_FEAT "${feat}" --CV_FOLDS "${CV_FOLDS}" --CV_FOLD_IDX "${fold}" \
              --WANDB_GROUP "${wandb_group}" --RUN_NAME "${run_name}" 2>&1 | tee "${run_log}"

            run_dir="$(sed -n 's/^Run directory: //p' "${run_log}" | head -1)"
            if [[ -n "${run_dir}" && -d "${run_dir}" ]]; then
              mv "${run_log}" "${run_dir}/output.log"
              chmod 0644 "${run_dir}/output.log"   # mktemp defaults to 0600
            else
              echo "WARNING: could not locate run directory for ${run_name}; learning curve will be missing" >&2
              rm -f "${run_log}"
            fi

            executed=$((executed + 1))
          done
        done

        combo_id=$((combo_id + 1))
      done
    done
  done
done

printf "\nTotal: %d | Executed: %d | Skipped: %d\n" "$total" "$executed" "$skipped"
