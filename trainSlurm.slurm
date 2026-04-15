#!/bin/bash
#
#SBATCH --partition=skylake_0096
#SBATCH -J gnn-comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --qos=skylake_0096



python pool_codes.py > resultsMCC/pool_k5.txt
