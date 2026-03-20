#!/bin/bash
#SBATCH --job-name=g_agg_results
#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -e

echo "======================================================================="
echo "   GENERIC – FINAL RESULTS AGGREGATION"
echo "======================================================================="

module --force purge > /dev/null 2>&1
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0
export PYTHONPATH="$PWD:$PYTHONPATH"

source generic_env/bin/activate
python3 -u pipeline_core/aggregate_results.py

echo "-----------------------------------------------------------------------"
echo "Final results aggregation complete!"
echo "All combined SOD and FTLE fields saved inside:"
echo "   ${OUTPUT_BASE_FOLDER}/final_results/"
echo "======================================================================="

