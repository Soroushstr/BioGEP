#!/bin/bash
# submit_loo_matrix.sh
#
# Submits 16 GPU jobs in parallel:
#   8 × LOO pooled training   (loo_pooled_v7.py   --held_out  SPECIES)
#   8 × single-species matrix (species_matrix_v7.py --train_species SPECIES)
#
# Then submits 1 heatmap job that runs after all 16 complete.
#
# Usage:  bash submit_loo_matrix.sh
#
# Output directories:
#   resultsMCC/          — all CSV results
#   loo_models/          — saved LOO models
#   single_species_models/ — saved single-species models
# ---------------------------------------------------------------------------

set -euo pipefail

SRC="/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
SPECIES=(elegans melanogaster musculus maripaludis bacillus sapiens arabidopsis saccharomyces)

# Slurm resource defaults — tune to your cluster's queue
PARTITION="gpu"
GRES="gpu:1"
CPUS=4
MEM="32G"

# Musculus has 49 K graphs and is the bottleneck; other species are much faster.
# We give everyone the same generous wall-time so the heatmap job can use a
# simple --dependency=afterok:... without worrying about stragglers.
WALLTIME_POOL="12:00:00"     # LOO folds (~3–4 h each, musculus held-in so ~3 h)
WALLTIME_SINGLE="08:00:00"   # single-species training (musculus is the longest ~4 h)
WALLTIME_PLOT="00:30:00"     # heatmap plotting

echo "================================================================"
echo "Submitting LOO + single-species matrix jobs"
echo "================================================================"

mkdir -p "${SRC}/slurm_logs"

JOB_IDS=()

# ---------------------------------------------------------------------------
# 1. LOO pooled training — 8 jobs
# ---------------------------------------------------------------------------
echo ""
echo "--- LOO pooled training (8 folds) ---"
for sp in "${SPECIES[@]}"; do
    JOB_NAME="loo_${sp}"
    JID=$(sbatch \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --gres="${GRES}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${WALLTIME_POOL}" \
        --output="${SRC}/slurm_logs/${JOB_NAME}_%j.out" \
        --error="${SRC}/slurm_logs/${JOB_NAME}_%j.err" \
        --wrap="cd ${SRC} && python loo_pooled_v7.py --held_out ${sp}" \
        | awk '{print $NF}')
    JOB_IDS+=("${JID}")
    echo "  Submitted ${JOB_NAME}  →  job ${JID}"
done

# ---------------------------------------------------------------------------
# 2. Single-species training — 8 jobs
# ---------------------------------------------------------------------------
echo ""
echo "--- Single-species matrix (8 training species) ---"
for sp in "${SPECIES[@]}"; do
    JOB_NAME="single_${sp}"
    JID=$(sbatch \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --gres="${GRES}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${WALLTIME_SINGLE}" \
        --output="${SRC}/slurm_logs/${JOB_NAME}_%j.out" \
        --error="${SRC}/slurm_logs/${JOB_NAME}_%j.err" \
        --wrap="cd ${SRC} && python species_matrix_v7.py --train_species ${sp}" \
        | awk '{print $NF}')
    JOB_IDS+=("${JID}")
    echo "  Submitted ${JOB_NAME}  →  job ${JID}"
done

# ---------------------------------------------------------------------------
# 3. Heatmap plotting — runs after ALL 16 jobs complete
# ---------------------------------------------------------------------------
DEP=$(IFS=:; echo "afterok:${JOB_IDS[*]}")   # afterok:ID1:ID2:...:ID16

echo ""
echo "--- Heatmap plotting (runs after all 16 jobs) ---"
PLOT_JID=$(sbatch \
    --job-name="plot_heatmap" \
    --partition="${PARTITION}" \
    --gres="${GRES}" \
    --cpus-per-task=2 \
    --mem="8G" \
    --time="${WALLTIME_PLOT}" \
    --dependency="${DEP}" \
    --output="${SRC}/slurm_logs/plot_heatmap_%j.out" \
    --error="${SRC}/slurm_logs/plot_heatmap_%j.err" \
    --wrap="cd ${SRC} && python plot_heatmap.py --metric both" \
    | awk '{print $NF}')
echo "  Submitted plot_heatmap  →  job ${PLOT_JID}"

echo ""
echo "================================================================"
echo "All jobs submitted."
echo ""
echo "  16 training jobs : ${JOB_IDS[*]}"
echo "  Heatmap job      : ${PLOT_JID}"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Outputs:       ${SRC}/resultsMCC/"
echo "================================================================"
