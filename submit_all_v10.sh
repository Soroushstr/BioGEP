#!/bin/bash
# submit_all_v10.sh — 24 jobs: constant-GRL + AUC stopping
#
# Fix: grl_alpha=1.0 (constant) means adversarial signal is present from
# epoch 1, preventing training-species specialisation during the critical
# early window where the model generalises best.
#
# Variants tested:
#   a05 = lambda_adv 0.5  (moderate adversarial)
#   a10 = lambda_adv 1.0  (strong adversarial)
#   a20 = lambda_adv 2.0  (very strong adversarial)
#
# Output:
#   resultsMCC/loo_v10_{a05|a10|a20}_{species}_results.csv  (24 files)
#   resultsMCC/loo_v10_{a05|a10|a20}_{species}.txt          (logs)
#
# Usage:
#   bash submit_all_v10.sh

set -e
cd "$(dirname "$0")"

SPECIES=(arabidopsis bacillus elegans maripaludis melanogaster musculus saccharomyces sapiens)
TAGS=(a05 a10 a20)

submitted=0
for sp in "${SPECIES[@]}"; do
  for tag in "${TAGS[@]}"; do
    slurm_file="loo_v10_${tag}_${sp}.slurm"
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
echo "Submitted $submitted / 24 jobs."
echo "Monitor : squeue -u \$USER"
echo "Results : python compare_v10_results.py"
echo "========================================"
