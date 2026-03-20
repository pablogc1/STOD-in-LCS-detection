#!/bin/bash
# ==============================================================================
#           TRAJECTORY EXPORT SCRIPT - Generates trajectories only
# ==============================================================================
# This script runs ONLY trajectory generation (Stage 1) for any system.
# Use this when you want to export trajectories for visualization/analysis.
#
# Usage:
#   ./run_trajectory_export.sh configs/trajectory_export_template.yaml
#
# After completion, trajectories will be in:
#   <output_base_folder>/trajectory_cache/partial/chunk_XXXX.h5
#
# You can then run the merge script to combine them:
#   python3 pipeline_core/merge_trajectories.py <output_base_folder>
# ==============================================================================

set -e
set -o pipefail

# --- 1. VALIDATE INPUT ---
if [ -z "$1" ]; then
    echo "ERROR: You must provide a path to a config file." >&2
    echo "Usage: ./run_trajectory_export.sh configs/your_config.yaml" >&2
    exit 1
fi
export SYSTEM_CONFIG_FILE=$(realpath "$1")

echo "======================================================================="
echo "   TRAJECTORY EXPORT PIPELINE"
echo "   Configuration: ${SYSTEM_CONFIG_FILE}"
echo "======================================================================="

# --- 2. ENVIRONMENT SETUP ---
echo "Setting up environment..."
module --force purge > /dev/null 2>&1 || true

# Load Modules (Adjusted to your cluster's specific HDF5 version)
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0
module load HDF5/1.14.0-gompi-2022b

# Activate Python Virtual Env
if [[ ! -f "generic_env/bin/activate" ]]; then
  echo "ERROR: virtualenv not found at generic_env/bin/activate" >&2
  exit 1
fi
source generic_env/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# --- 3. COMPILE C++ BACKEND (if needed) ---
echo "-----------------------------------------------------------------------"
echo "Checking C++ Backend..."
if [[ -d "cpp_backend" ]]; then
    if [[ ! -f "cpp_backend/gen_traj" ]] || [[ "cpp_backend/gen_traj.cpp" -nt "cpp_backend/gen_traj" ]]; then
        echo "Compiling C++ Backend..."
        cd cpp_backend
        make clean > /dev/null
        make gen_traj
        if [[ $? -ne 0 ]]; then
            echo "!!! C++ COMPILATION FAILED !!!"
            exit 1
        fi
        cd ..
        echo "  -> Compilation successful."
    else
        echo "  -> C++ binary is up to date."
    fi
else
    echo "ERROR: 'cpp_backend' directory not found!"
    exit 1
fi

# --- 4. CALCULATE JOB ARRAY SIZE ---
echo "-----------------------------------------------------------------------"
echo "Calculating job configuration..."
eval $(python3 -c "
import yaml
cfg=yaml.safe_load(open('${SYSTEM_CONFIG_FILE}'))
grid_y = cfg['grid_params']['grid_resolution_y']
grid_x = cfg['grid_params']['grid_resolution_x']
max_jobs = cfg['hpc_params']['max_desired_jobs']
num_jobs = min(max_jobs, grid_y)
total_cells = grid_x * grid_y
t_end = cfg['time_params']['total_integration_time_end']
dt = cfg['time_params']['time_step_dt']
n_steps = int(t_end / dt) + 1
field_mode = cfg.get('field_mode', 'standard')
print(f'export NUM_ARRAY_JOBS_FINAL={num_jobs}')
print(f'export OUTPUT_BASE_FOLDER=\"{cfg[\"output_base_folder\"]}\"')
print(f'export SLURM_LOG_FOLDER=\"{cfg[\"slurm_log_folder\"]}\"')
print(f'export GEN_CORES={cfg[\"hpc_params\"][\"num_cores_per_job_generate\"]}')
print(f'export TOTAL_GRID_CELLS={total_cells}')
print(f'export GRID_X={grid_x}')
print(f'export GRID_Y={grid_y}')
print(f'export N_TIME_STEPS={n_steps}')
print(f'export FIELD_MODE=\"{field_mode}\"')
print(f'export SYSTEM_MODULE=\"{cfg[\"system_module\"]}\"')
")

echo "  System:         ${SYSTEM_MODULE}"
echo "  Grid:           ${GRID_X} × ${GRID_Y} = ${TOTAL_GRID_CELLS} cells"
echo "  Time steps:     ${N_TIME_STEPS}"
echo "  Field mode:     ${FIELD_MODE}"
echo "  Parallel jobs:  ${NUM_ARRAY_JOBS_FINAL}"
echo "-----------------------------------------------------------------------"

# --- 5. PREPARE DIRECTORIES ---
export TRAJECTORY_FOLDER="${OUTPUT_BASE_FOLDER}/trajectory_cache/partial"
mkdir -p "${SLURM_LOG_FOLDER}" "${TRAJECTORY_FOLDER}"

# Clean old log files
rm -f "${SLURM_LOG_FOLDER}"/gen_traj_*.out "${SLURM_LOG_FOLDER}"/gen_traj_*.err 2>/dev/null || true

# --- 6. SUBMIT TRAJECTORY GENERATION JOBS ---
echo ""
echo ">>> Submitting trajectory generation jobs..."

# Handle field mode override for orthogonal
if [ "${FIELD_MODE}" = "orthogonal" ]; then
    export FIELD_MODE_OVERRIDE="orthogonal"
else
    export FIELD_MODE_OVERRIDE="standard"
fi

JID_GEN_TRAJ=$(sbatch --parsable \
       --export=ALL \
       --array=1-${NUM_ARRAY_JOBS_FINAL} \
       --output=${SLURM_LOG_FOLDER}/gen_traj_%A_%a.out \
       --error=${SLURM_LOG_FOLDER}/gen_traj_%A_%a.err \
       --cpus-per-task=${GEN_CORES} \
       pipeline_core/sbatch_generate_traj.slurm)

echo "    Submitted Job ID: $JID_GEN_TRAJ"
echo "    Logs: ${SLURM_LOG_FOLDER}/gen_traj_${JID_GEN_TRAJ}_*.out"
echo ""
echo ">>> Waiting for jobs to complete..."

# --- 7. WAIT FOR COMPLETION ---
while true; do
    set +e
    RUNNING_JOBS=$(squeue -h -j "$JID_GEN_TRAJ" 2>/dev/null | wc -l)
    SQUEUE_EXIT=$?
    set -e
    if [ "$SQUEUE_EXIT" -ne 0 ] || [ "$RUNNING_JOBS" -eq 0 ]; then
        break
    fi
    
    # Simple progress indicator
    COMPLETED=$(ls -1 "${TRAJECTORY_FOLDER}"/chunk_*.h5 2>/dev/null | wc -l || echo 0)
    echo "    Jobs running: ${RUNNING_JOBS}, Files completed: ${COMPLETED}/${NUM_ARRAY_JOBS_FINAL}"
    sleep 30
done

# --- 8. VERIFY RESULTS ---
echo ""
echo "-----------------------------------------------------------------------"
COMPLETED_FILES=$(ls -1 "${TRAJECTORY_FOLDER}"/chunk_*.h5 2>/dev/null | wc -l || echo 0)
echo ">>> Trajectory generation complete!"
echo "    Files generated: ${COMPLETED_FILES}/${NUM_ARRAY_JOBS_FINAL}"
echo "    Location: ${TRAJECTORY_FOLDER}"
echo ""

if [ "$COMPLETED_FILES" -lt "$NUM_ARRAY_JOBS_FINAL" ]; then
    echo "WARNING: Some jobs may have failed. Check logs in ${SLURM_LOG_FOLDER}"
fi

# --- 9. MERGE TRAJECTORY FILES ---
echo "-----------------------------------------------------------------------"
echo ">>> Merging trajectory files into single HDF5..."
python3 pipeline_core/merge_trajectories.py "${OUTPUT_BASE_FOLDER}" --config "${SYSTEM_CONFIG_FILE}"

if [ $? -eq 0 ]; then
    MERGED_FILE="${OUTPUT_BASE_FOLDER}/trajectories_merged.h5"
    if [ -f "$MERGED_FILE" ]; then
        FILE_SIZE=$(du -h "$MERGED_FILE" | cut -f1)
        echo ""
        echo "======================================================================="
        echo "   TRAJECTORY EXPORT COMPLETE"
        echo "======================================================================="
        echo ""
        echo "  Merged file: ${MERGED_FILE}"
        echo "  File size:   ${FILE_SIZE}"
        echo ""
        echo "  Download this file and use plot_trajectories_spyder.py to visualize!"
        echo "======================================================================="
    fi
else
    echo "WARNING: Merge failed. You can try manually:"
    echo "  python3 pipeline_core/merge_trajectories.py ${OUTPUT_BASE_FOLDER}"
fi

