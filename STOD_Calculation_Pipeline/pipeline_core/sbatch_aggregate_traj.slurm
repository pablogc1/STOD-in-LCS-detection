#!/bin/bash
#SBATCH --job-name=g_agg_traj
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

set -e

echo "--- Starting Generic Trajectory Aggregation Job ---"
echo "System Config File: ${SYSTEM_CONFIG_FILE}"

module --force purge > /dev/null 2>&1
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0
export PYTHONPATH="$PWD:$PYTHONPATH"

source generic_env/bin/activate
python3 -u pipeline_core/aggregate_trajectories.py
echo "--- Generic Aggregation Finished Successfully ---"

