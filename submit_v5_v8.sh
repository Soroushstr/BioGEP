#!/bin/bash
# Submit all 4 new runs as independent slurm jobs.
# Each job writes its output to resultsMCC/cfix_10..13.txt
#
# Usage: bash submit_v5_v8.sh
# Or submit one at a time: sbatch run_v5.sh

SRC="/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
RESULTS="$SRC/resultsMCC"

# ── helper: write a slurm script and submit it ──────────────────────────────
submit_job() {
    local VNAME=$1     # e.g. v5
    local CFIX=$2      # e.g. cfix_10
    local SCRIPT="$SRC/run_${VNAME}.sh"

    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH -J GEP_${VNAME}
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH -o $SRC/slurm_${VNAME}_%j.out
#SBATCH -e $SRC/slurm_${VNAME}_%j.err
#SBATCH --time=12:00:00

module purge
module load cuda/11.8

cd $SRC
python ara_ara_code_${VNAME}.py > $RESULTS/${CFIX}.txt 2>&1
EOF

    echo "Submitting ${VNAME} → ${CFIX}.txt"
    sbatch "$SCRIPT"
}

mkdir -p "$RESULTS"

submit_job v5 cfix_10
submit_job v6 cfix_11
submit_job v7 cfix_12
submit_job v8 cfix_13

echo "All 4 jobs submitted. Monitor with: squeue -u \$USER"
