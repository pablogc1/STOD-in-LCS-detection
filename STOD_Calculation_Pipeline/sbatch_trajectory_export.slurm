#!/bin/bash
#SBATCH --job-name=traj_export_master
#SBATCH --output=traj_export_master_%j.out
#SBATCH --error=traj_export_master_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
# ==============================================================================
#           TRAJECTORY EXPORT - SLURM MASTER SCRIPT
# ==============================================================================
# IMPORTANT: Before first run, create the log directory:
#   mkdir -p slurm_logs_trajectory_export
#
# Submit this script with:
#   sbatch sbatch_trajectory_export.slurm configs/trajectory_export_template.yaml
#
# This will:
#   1. Submit trajectory generation jobs (array)
#   2. Submit merge job (runs after generation completes)
#   3. You can monitor progress via squeue and log files
# ==============================================================================

set -e
set -o pipefail

# --- 1. VALIDATE INPUT ---
if [ -z "$1" ]; then
    echo "ERROR: You must provide a path to a config file." >&2
    echo "Usage: sbatch sbatch_trajectory_export.slurm configs/your_config.yaml" >&2
    exit 1
fi
export SYSTEM_CONFIG_FILE=$(realpath "$1")

echo "======================================================================="
echo "   TRAJECTORY EXPORT PIPELINE"
echo "   Configuration: ${SYSTEM_CONFIG_FILE}"
echo "   Started: $(date)"
echo "======================================================================="

# --- 2. ENVIRONMENT SETUP ---
echo "Setting up environment..."
module --force purge > /dev/null 2>&1 || true
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
        make clean > /dev/null 2>&1 || true
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

# --- 4. CALCULATE JOB CONFIGURATION ---
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
print(f'export NUM_ARRAY_JOBS={num_jobs}')
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
echo "  Parallel jobs:  ${NUM_ARRAY_JOBS}"
echo "-----------------------------------------------------------------------"

# --- 5. PREPARE DIRECTORIES ---
export TRAJECTORY_FOLDER="${OUTPUT_BASE_FOLDER}/trajectory_cache/partial"
mkdir -p "${SLURM_LOG_FOLDER}" "${TRAJECTORY_FOLDER}"

# Clean old log files
rm -f "${SLURM_LOG_FOLDER}"/gen_traj_*.out "${SLURM_LOG_FOLDER}"/gen_traj_*.err 2>/dev/null || true
rm -f "${SLURM_LOG_FOLDER}"/merge_*.out "${SLURM_LOG_FOLDER}"/merge_*.err 2>/dev/null || true

# --- 6. SET FIELD MODE ---
if [ "${FIELD_MODE}" = "orthogonal" ]; then
    export FIELD_MODE_OVERRIDE="orthogonal"
else
    export FIELD_MODE_OVERRIDE="standard"
fi

# --- 7. SUBMIT TRAJECTORY GENERATION JOBS ---
echo ""
echo ">>> STAGE 1: Submitting trajectory generation jobs..."

JID_GEN=$(sbatch --parsable \
       --export=ALL \
       --array=1-${NUM_ARRAY_JOBS} \
       --output=${SLURM_LOG_FOLDER}/gen_traj_%A_%a.out \
       --error=${SLURM_LOG_FOLDER}/gen_traj_%A_%a.err \
       --cpus-per-task=${GEN_CORES} \
       pipeline_core/sbatch_generate_traj.slurm)

echo "    Trajectory Job Array ID: $JID_GEN"
echo "    Logs: ${SLURM_LOG_FOLDER}/gen_traj_${JID_GEN}_*.out"

# --- 8. SUBMIT MERGE JOB (depends on trajectory jobs) ---
echo ""
echo ">>> STAGE 2: Submitting merge job (will run after trajectories complete)..."

JID_MERGE=$(sbatch --parsable \
       --dependency=afterok:${JID_GEN} \
       --export=ALL \
       --output=${SLURM_LOG_FOLDER}/merge_%j.out \
       --error=${SLURM_LOG_FOLDER}/merge_%j.err \
       pipeline_core/sbatch_merge_trajectories.slurm)

echo "    Merge Job ID: $JID_MERGE"
echo "    Log: ${SLURM_LOG_FOLDER}/merge_${JID_MERGE}.out"

# --- 9. SUMMARY ---
echo ""
echo "======================================================================="
echo "   JOBS SUBMITTED SUCCESSFULLY"
echo "======================================================================="
echo ""
echo "  Trajectory Generation: Job $JID_GEN (${NUM_ARRAY_JOBS} tasks)"
echo "  Merge:                 Job $JID_MERGE (runs after generation)"
echo ""
echo "  Monitor progress:"
echo "    squeue -u \$USER"
echo "    tail -f ${SLURM_LOG_FOLDER}/gen_traj_${JID_GEN}_1.out"
echo ""
echo "  When complete, download:"
echo "    ${OUTPUT_BASE_FOLDER}/trajectories_merged.h5"
echo ""
echo "======================================================================="

