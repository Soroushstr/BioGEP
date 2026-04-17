#!/bin/bash
#SBATCH -J loo_v10_a10_sapiens
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

cd /gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src
python loo_pooled_v10.py --held_out sapiens --lambda_adv 1.0 > resultsMCC/loo_v10_a10_sapiens.txt
