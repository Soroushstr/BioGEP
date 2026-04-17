#!/bin/bash
#SBATCH -J loo_v9_lv20_bacillus
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

cd /gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src
python loo_pooled_v9.py --held_out bacillus --val_frac 0.20 > resultsMCC/loo_v9_lv20_bacillus.txt
