#!/bin/bash
#SBATCH --job-name=g_worker
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G

set -e

echo "======================================================================="
echo "Starting Generic Worker Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "======================================================================="

module --force purge > /dev/null 2>&1
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# --- ADD THIS LINE ---
module load HDF5/1.14.0-gompi-2022b
# ---------------------

export PYTHONPATH="$PWD:$PYTHONPATH"

source generic_env/bin/activate

python3 -u pipeline_core/run_worker.py \
    --worker_id ${SLURM_ARRAY_TASK_ID}

echo "======================================================================="
echo "Worker Task Finished."
