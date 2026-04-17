#!/bin/bash
# submit_all_v9.sh — Launch all 16 LOO-val Priority-1 experiments in parallel
#
# Experiment: LOO-val early stopping (Priority 1 fix)
#   lv10 = 10% of held-out species as val set (90% for test)
#   lv20 = 20% of held-out species as val set (80% for test)
#
# Expected output:
#   resultsMCC/loo_v9_lv10_{species}_results.csv  (8 files)
#   resultsMCC/loo_v9_lv20_{species}_results.csv  (8 files)
#   resultsMCC/loo_v9_lv10_{species}.txt          (logs)
#   resultsMCC/loo_v9_lv20_{species}.txt          (logs)
#
# Usage:
#   bash submit_all_v9.sh

set -e
cd "$(dirname "$0")"

SPECIES=(arabidopsis bacillus elegans maripaludis melanogaster musculus saccharomyces sapiens)
VAL_FRACS=(lv10 lv20)

submitted=0
for sp in "${SPECIES[@]}"; do
  for vf in "${VAL_FRACS[@]}"; do
    slurm_file="loo_v9_${vf}_${sp}.slurm"
    if [ ! -f "$slurm_file" ]; then
      echo "[SKIP] $slurm_file not found"
      continue
    fi
    job_id=$(sbatch "$slurm_file" | awk '{print $NF}')
    echo "Submitted $slurm_file → job $job_id"
    submitted=$((submitted + 1))
  done
done

echo ""
echo "========================================"
echo "Submitted $submitted jobs total."
echo "Monitor with: squeue -u \$USER"
echo ""
echo "When all jobs finish, compare results:"
echo "  python compare_v9_results.py"
echo "========================================"
