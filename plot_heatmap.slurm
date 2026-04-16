#!/bin/bash
#
#  usage: sbatch plot_heatmap.slurm
#  Run this after all 16 LOO + single-species jobs finish.
#
#SBATCH -J plot_heatmap
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:1

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

cd /gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src
python plot_heatmap.py --metric both > resultsMCC/heatmap.txt
