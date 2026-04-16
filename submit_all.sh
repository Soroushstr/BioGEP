#!/bin/bash
# submit_all.sh — submit all 16 LOO + single-species jobs, then the heatmap job
# Usage: bash submit_all.sh

cd /gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src

echo "Submitting LOO jobs..."
sbatch loo_elegans.slurm
sbatch loo_melanogaster.slurm
sbatch loo_musculus.slurm
sbatch loo_maripaludis.slurm
sbatch loo_bacillus.slurm
sbatch loo_sapiens.slurm
sbatch loo_arabidopsis.slurm
sbatch loo_saccharomyces.slurm

echo "Submitting single-species jobs..."
sbatch single_elegans.slurm
sbatch single_melanogaster.slurm
sbatch single_musculus.slurm
sbatch single_maripaludis.slurm
sbatch single_bacillus.slurm
sbatch single_sapiens.slurm
sbatch single_arabidopsis.slurm
sbatch single_saccharomyces.slurm

echo "Done. Monitor with: squeue -u \$USER"
echo "Once all 16 jobs finish, run: sbatch plot_heatmap.slurm"
