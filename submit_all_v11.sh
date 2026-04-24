#!/bin/bash
# submit_all_v11.sh — 32 jobs: phylogeny-aware LOO training
#
# v10 diagnostic: constant GRL gave mean MCC 0.1768 vs v9 0.1738.
# Best epochs still 1-5 for 6/8 species at lambda=2.0.
# Decision rule: GRL insufficient → Priority 2 (phylogeny-aware training).
#
# Two variants:
#   hard  (8 jobs) : exclude cross-superkingdom training species, no GRL
#   soft  (24 jobs): weight loss by exp(-beta*phylo_dist), GRL kept
#                    beta ∈ {0.5, 1.0, 2.0}  (tags: b05, b10, b20)
#
# Phylo groups: metazoa(4) | fungi(1) | plantae(1) | bacteria(1) | archaea(1)
# Hard training sets per held-out:
#   arabidopsis  → saccharomyces only (1 species)
#   bacillus     → maripaludis only   (1 species)
#   elegans      → mel+mus+sap+sac    (4 species)
#   maripaludis  → bacillus only      (1 species)
#   melanogaster → ele+mus+sap+sac    (4 species)
#   musculus     → ele+mel+sap+sac    (4 species)
#   saccharomyces→ ele+mel+mus+sap+ara(5 species)
#   sapiens      → ele+mel+mus+sac    (4 species)
#
# Output:
#   resultsMCC/loo_v11_hard_{species}_results.csv         (8 files)
#   resultsMCC/loo_v11_soft_b{05|10|20}_{species}_results.csv (24 files)
#
# Usage:
#   bash submit_all_v11.sh

set -e
cd "$(dirname "$0")"

SPECIES=(arabidopsis bacillus elegans maripaludis melanogaster musculus saccharomyces sapiens)
SLURM_PARTITION="zen3_0512_a100x2"
SLURM_QOS="zen3_0512_a100x2"
CUDA_MODULE="cuda/11.8.0-gcc-12.2.0-bplw5nu"
SRC="/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"

submitted=0

# ── Hard exclusion (8 jobs, no GRL) ─────────────────────────────────────────
echo "=== Submitting v11 hard (8 jobs) ==="
for sp in "${SPECIES[@]}"; do
  job_name="loo_v11_hard_${sp}"
  log_file="${SRC}/resultsMCC/loo_v11_hard_${sp}.txt"

  job_id=$(sbatch \
    --job-name="$job_name" \
    --nodes=1 \
    --partition="$SLURM_PARTITION" \
    --qos="$SLURM_QOS" \
    --gres=gpu:2 \
    --wrap="module purge; module load ${CUDA_MODULE}; cd ${SRC}; python loo_pooled_v11.py --held_out ${sp} --phylo_mode hard > ${log_file} 2>&1" \
    | awk '{print $NF}')
  echo "  Submitted $job_name → job $job_id"
  submitted=$((submitted + 1))
done

# ── Soft weighting (24 jobs, GRL on) ────────────────────────────────────────
echo ""
echo "=== Submitting v11 soft (24 jobs) ==="
declare -A BETA_TAGS=( ["0.5"]="b05" ["1.0"]="b10" ["2.0"]="b20" )
for beta in 0.5 1.0 2.0; do
  btag="${BETA_TAGS[$beta]}"
  for sp in "${SPECIES[@]}"; do
    job_name="loo_v11_soft_${btag}_${sp}"
    log_file="${SRC}/resultsMCC/loo_v11_soft_${btag}_${sp}.txt"

    job_id=$(sbatch \
      --job-name="$job_name" \
      --nodes=1 \
      --partition="$SLURM_PARTITION" \
      --qos="$SLURM_QOS" \
      --gres=gpu:2 \
      --wrap="module purge; module load ${CUDA_MODULE}; cd ${SRC}; python loo_pooled_v11.py --held_out ${sp} --phylo_mode soft --phylo_beta ${beta} > ${log_file} 2>&1" \
      | awk '{print $NF}')
    echo "  Submitted $job_name → job $job_id"
    submitted=$((submitted + 1))
  done
done

echo ""
echo "========================================"
echo "Submitted $submitted / 32 jobs."
echo "Monitor : squeue -u \$USER"
echo "Results : python compare_v11_results.py"
echo "========================================"
