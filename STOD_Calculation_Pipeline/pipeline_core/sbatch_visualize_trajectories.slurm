#!/bin/bash
#SBATCH --job-name=TRAJ_VIZ_ARRAY
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_viz_%A_%a.out
#SBATCH --error=slurm_viz_%A_%a.err

set -euo pipefail

# --- Environment setup ---
module --force purge >/dev/null 2>&1 || true
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

if [[ ! -f "generic_env/bin/activate" ]]; then
  echo "ERROR: virtualenv not found" >&2
  exit 1
fi
source generic_env/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# --- Parameters ---
CFG=${CFG:-""}
OUT=${OUT:-"trajectory_evolution_output"}
GRID=${GRID:-20}
SNAPS=${SNAPS:-15}
TIME=${TIME:-""}
DT=${DT:-""}
MAX_POINTS=${MAX_POINTS:-100}

if [[ -z "$CFG" ]]; then
    echo "ERROR: CFG variable not set."
    exit 1
fi

# Determine if we are running a specific frame from the array
FRAME_ARG=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    FRAME_ARG="--frame_idx $SLURM_ARRAY_TASK_ID"
    echo "Processing Frame: $SLURM_ARRAY_TASK_ID"
fi

echo "Running Visualization Task..."
CMD="python3 pipeline_core/visualize_trajectories.py \
    --config $CFG \
    --output_dir $OUT \
    --grid_size $GRID \
    --n_snapshots $SNAPS \
    --max_points $MAX_POINTS \
    $FRAME_ARG"

[[ -n "$TIME" ]] && CMD="$CMD --total_time $TIME"
[[ -n "$DT" ]] && CMD="$CMD --dt $DT"

$CMD

echo "Task Complete."

