#!/bin/bash
#SBATCH -J loo_v8_musculus
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

cd /gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src
python loo_pooled_v8.py --held_out musculus > resultsMCC/loo_v8_musculus.txt
