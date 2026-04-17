#!/bin/bash
# Submit all 8 LOO v8 jobs in parallel.
# Results go to resultsMCC/loo_v8_{species}.txt and loo_v8_{species}_results.csv

set -e
cd "$(dirname "$0")"

for sp in elegans melanogaster musculus maripaludis bacillus sapiens arabidopsis saccharomyces; do
    jid=$(sbatch --parsable loo_v8_${sp}.slurm)
    echo "Submitted loo_v8_${sp}: job ${jid}"
done

echo ""
echo "All 8 v8 LOO jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Results will appear in resultsMCC/loo_v8_*"
