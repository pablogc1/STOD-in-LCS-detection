#!/bin/bash
#SBATCH --job-name=g_gen_traj
#SBATCH --time=08:00:00
#SBATCH --mem=32G

set -e

echo "======================================================================="
echo "Starting Generic Trajectory Generation Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "======================================================================="

module --force purge > /dev/null 2>&1
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# --- ADD THIS LINE ---
module load HDF5/1.14.0-gompi-2022b
# ---------------------

export PYTHONPATH="$PWD:$PYTHONPATH"

source generic_env/bin/activate

# Optional: Print debug info to confirm library is found
# echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"

python3 -u pipeline_core/generate_trajectories.py \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --total_jobs ${SLURM_ARRAY_TASK_COUNT} \
    --num_cores ${SLURM_CPUS_PER_TASK}

echo "======================================================================="
echo "Task Finished."


