#!/bin/bash
# ==============================================================================
#           GENERIC C++ SOD/FTLE PIPELINE - MASTER RUN SCRIPT
# ==============================================================================

set -e
set -o pipefail

# Track if we're exiting normally (to distinguish from errors)
SCRIPT_EXIT_NORMAL=0

# Signal trap to handle script termination gracefully
cleanup_on_exit() {
    EXIT_CODE=$?
    # Only show warning if we were interrupted (non-zero exit) AND it's not a normal exit
    if [ $EXIT_CODE -ne 0 ] && [ "$SCRIPT_EXIT_NORMAL" -eq 0 ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [ERROR] Master script encountered an error (exit code: $EXIT_CODE)"
        if [ -n "${BASH_COMMAND:-}" ]; then
            echo "  Last command: ${BASH_COMMAND}"
        fi
        if [ -n "$JID_GEN_TRAJ" ]; then
            echo "  Trajectory generation jobs may still be running."
            echo "  Job ID: $JID_GEN_TRAJ"
            if [ -n "$SLURM_LOG_FOLDER" ]; then
                echo "  Check individual job logs in: ${SLURM_LOG_FOLDER}"
            fi
            echo "  Use 'squeue -j $JID_GEN_TRAJ' to check job status"
        else
            echo "  Script failed before job submission."
            echo "  Check the error messages above for details."
        fi
        echo ""
        echo "  TIP: Look at the output above this message to find the failing command."
        echo "  Common causes:"
        echo "    - Missing files or directories"
        echo "    - Python/YAML parsing errors"
        echo "    - SLURM command failures"
        echo "    - Missing dependencies (bc, awk, etc.)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
    exit $EXIT_CODE
}

trap cleanup_on_exit EXIT INT TERM

# Debug: Log memory usage periodically (helps diagnose OOM issues)
log_memory_usage() {
    if command -v free &> /dev/null; then
        echo "[DEBUG $(date +%H:%M:%S)] Memory: $(free -h 2>/dev/null | awk '/^Mem:/ {print $3 "/" $2}' || echo 'N/A')"
    fi
}

# --- 1. VALIDATE INPUT ---
if [ -z "$1" ]; then
    echo "ERROR: You must provide a path to a system config file." >&2
    exit 1
fi
export SYSTEM_CONFIG_FILE=$(realpath "$1")

echo "======================================================================="
echo "   STARTING C++ ACCELERATED PIPELINE"
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

# --- 3. COMPILE C++ BACKEND ---
echo "-----------------------------------------------------------------------"
echo "Compiling C++ Backend..."
if [[ -d "cpp_backend" ]]; then
    cd cpp_backend
    make clean > /dev/null
    make
    if [[ $? -ne 0 ]]; then
        echo "!!! C++ COMPILATION FAILED !!!"
        exit 1
    fi
    cd ..
else
    echo "ERROR: 'cpp_backend' directory not found!"
    exit 1
fi
echo "  -> Compilation successful."

# --- 4. RUN CANARY TESTS (VITAL) ---
echo "-----------------------------------------------------------------------"
echo "Running C++ SOD Logic Canary Test (Quick Validation)..."
# This ensures the struct/bool return types and math are valid before HPC jobs start
if ./cpp_backend/sod_canary; then
    echo ">>> C++ CANARY PASSED: Basic STOD logic is mathematically correct."
else
    echo ""
    echo "!!! C++ CANARY FAILED: Aborting Pipeline Immediately. !!!"
    exit 1
fi
echo "-----------------------------------------------------------------------"

echo "Running Python STOD/FINSTOD Canary Test (Full Demonstration)..."
# This demonstrates that the STOD logic functions work correctly for both STOD and FINSTOD
if python3 pipeline_core/sod_canary.py; then
    echo ">>> PYTHON CANARY PASSED: STOD/FINSTOD functions verified."
else
    echo ""
    echo "!!! PYTHON CANARY FAILED: Aborting Pipeline Immediately. !!!"
    exit 1
fi
echo "-----------------------------------------------------------------------"

# --- 5. CALCULATE JOB ARRAY SIZE AND DETECT FIELD MODE ---
echo "Calculating job array size and detecting field mode..."
eval $(python3 -c "
import yaml, math
cfg=yaml.safe_load(open('${SYSTEM_CONFIG_FILE}'))
grid_y = cfg['grid_params']['grid_resolution_y']
grid_x = cfg['grid_params']['grid_resolution_x']
max_jobs = cfg['hpc_params']['max_desired_jobs']
# Ensure at least 1 row per job, max_jobs cannot exceed grid_y
num_jobs = min(max_jobs, grid_y)
total_cells = grid_x * grid_y
field_mode = cfg.get('field_mode', 'standard')
system_module = cfg.get('system_module', '')
# Only LiCN is a true high-dimensional system where orthogonal mode is problematic
# Lorenz is now a 2D Poincaré section (z fixed), so it's fine with orthogonal mode
is_3d = 'licn' in system_module.lower()
print(f'export NUM_ARRAY_JOBS_FINAL={num_jobs}')
print(f'export OUTPUT_BASE_FOLDER_ORIG=\"{cfg[\"output_base_folder\"]}\"')
print(f'export SLURM_LOG_FOLDER_ORIG=\"{cfg[\"slurm_log_folder\"]}\"')
print(f'export GEN_CORES={cfg[\"hpc_params\"][\"num_cores_per_job_generate\"]}')
print(f'export WORKER_CORES={cfg[\"hpc_params\"][\"num_cores_per_job_worker\"]}')
print(f'export TOTAL_GRID_CELLS={total_cells}')
print(f'export GRID_X={grid_x}')
print(f'export GRID_Y={grid_y}')
print(f'export FIELD_MODE_CONFIG=\"{field_mode}\"')
print(f'export IS_3D_SYSTEM={str(is_3d).lower()}')
")

# Determine which field modes to run
if [ "${FIELD_MODE_CONFIG}" = "both" ]; then
    FIELD_MODES_TO_RUN="standard orthogonal"
    echo "  -> Field mode: BOTH (will run standard and orthogonal separately)"
else
    FIELD_MODES_TO_RUN="${FIELD_MODE_CONFIG}"
    echo "  -> Field mode: ${FIELD_MODE_CONFIG}"
fi

# Warning for truly high-dimensional systems with orthogonal mode
# Note: Lorenz is now a 2D Poincaré section (z fixed), so it's fine
# Only LiCN is a true 4D system where orthogonal mode is problematic
if [ "${IS_3D_SYSTEM}" = "true" ] && [ "${FIELD_MODE_CONFIG}" != "standard" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [WARNING] ORTHOGONAL MODE ON HIGH-DIMENSIONAL SYSTEM (LiCN)"
    echo "  The orthogonal field for 4D systems creates non-physical dynamics."
    echo "  Only the first two velocity components are rotated."
    echo "  This may cause very slow integration and meaningless results."
    echo "  Consider using 'standard' field_mode for LiCN."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

echo "  -> Pipeline will run with ${NUM_ARRAY_JOBS_FINAL} parallel chunks."
echo "-----------------------------------------------------------------------"

# ==============================================================================
# MAIN PIPELINE LOOP - Runs once for single mode, twice for "both" mode
# ==============================================================================
for CURRENT_FIELD_MODE in ${FIELD_MODES_TO_RUN}; do

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    PROCESSING FIELD MODE: ${CURRENT_FIELD_MODE}"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Export the field mode override for workers/generators to use
export FIELD_MODE_OVERRIDE="${CURRENT_FIELD_MODE}"

# --- 6. PREPARE DIRECTORIES ---
# For "both" mode, add field mode suffix to output folders
if [ "${FIELD_MODE_CONFIG}" = "both" ]; then
    export OUTPUT_BASE_FOLDER="${OUTPUT_BASE_FOLDER_ORIG}/${CURRENT_FIELD_MODE}"
    export SLURM_LOG_FOLDER="${SLURM_LOG_FOLDER_ORIG}_${CURRENT_FIELD_MODE}"
else
    export OUTPUT_BASE_FOLDER="${OUTPUT_BASE_FOLDER_ORIG}"
    export SLURM_LOG_FOLDER="${SLURM_LOG_FOLDER_ORIG}"
fi

export TRAJECTORY_FOLDER="${OUTPUT_BASE_FOLDER}/trajectory_cache/partial"
export RESULTS_FOLDER="${OUTPUT_BASE_FOLDER}/partial_results"
export FINAL_FOLDER="${OUTPUT_BASE_FOLDER}/final_results"

echo "  Output directories:"
echo "    Base:       ${OUTPUT_BASE_FOLDER}"
echo "    Logs:       ${SLURM_LOG_FOLDER}"
echo "    Trajectories: ${TRAJECTORY_FOLDER}"
echo "    Results:    ${RESULTS_FOLDER}"
echo "    Final:      ${FINAL_FOLDER}"
echo ""

mkdir -p "${SLURM_LOG_FOLDER}" "${TRAJECTORY_FOLDER}" "${RESULTS_FOLDER}" "${FINAL_FOLDER}"

# Clean old log files to prevent accumulation across runs
echo "Cleaning old log files..."
rm -f "${SLURM_LOG_FOLDER}"/gen_traj_*.out "${SLURM_LOG_FOLDER}"/gen_traj_*.err 2>/dev/null || true
rm -f "${SLURM_LOG_FOLDER}"/worker_*.out "${SLURM_LOG_FOLDER}"/worker_*.err 2>/dev/null || true
rm -f "${SLURM_LOG_FOLDER}"/agg_*.out "${SLURM_LOG_FOLDER}"/agg_*.err 2>/dev/null || true

# Create dummy HDF5 file for global metrics (if needed) - do this once before workers start
DUMMY_INPUT="${OUTPUT_BASE_FOLDER}/trajectory_cache/dummy.h5"
if [ ! -f "${DUMMY_INPUT}" ]; then
    echo "Creating dummy HDF5 file for global metrics..."
    python3 << EOF
import h5py
import numpy as np
import yaml

with open('${SYSTEM_CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

gp = config['grid_params']
tp = config['time_params']
nx = int(gp['grid_resolution_x'])
dt = tp.get('time_step_dt', 0.1)
t_total = tp['total_integration_time_end']
n_steps = max(2, int(t_total / dt) + 1)

import os
os.makedirs(os.path.dirname('${DUMMY_INPUT}'), exist_ok=True)
with h5py.File('${DUMMY_INPUT}', 'w') as f:
    f.create_dataset('forward', shape=(1, nx, n_steps, 2), dtype=np.float64)
EOF
    echo "  -> Dummy file created: ${DUMMY_INPUT}"
fi

# ==============================================================================
# STAGE 1: Trajectory Generation (C++ Backend)
# ==============================================================================
echo ""
echo ">>> STAGE 1: Generating Trajectories (C++) [${CURRENT_FIELD_MODE} field]..."
JID_GEN_TRAJ=$(sbatch --parsable \
       --export=ALL \
       --array=1-${NUM_ARRAY_JOBS_FINAL} \
       --output=${SLURM_LOG_FOLDER}/gen_traj_%A_%a.out \
       --error=${SLURM_LOG_FOLDER}/gen_traj_%A_%a.err \
       --cpus-per-task=${GEN_CORES} \
       pipeline_core/sbatch_generate_traj.slurm)

echo "    Submitted Job ID: $JID_GEN_TRAJ"
echo "    Monitoring progress (checking logs every 10 seconds)..."
echo "    Total grid cells: ${TOTAL_GRID_CELLS} (${GRID_X} × ${GRID_Y})"
echo ""

# Progress tracking variables
START_TIME=$(date +%s)
LAST_PROGRESS_TIME=$START_TIME
LAST_TOTAL_PROCESSED=0
FIRST_ITERATION=1

# Declare associative array for job completion status (outside loop for compatibility)
declare -A JOB_COMPLETE_STATUS

# Wait for all array tasks to actually finish
echo "    Waiting for all ${NUM_ARRAY_JOBS_FINAL} trajectory generation jobs to finish..."
echo ""

while true; do
    # Check if any jobs from this array are still running
    # Use set +e temporarily to handle squeue failures gracefully
    set +e
    RUNNING_JOBS=$(squeue -h -j "$JID_GEN_TRAJ" 2>/dev/null | wc -l)
    SQUEUE_EXIT=$?
    set -e
    if [ "$SQUEUE_EXIT" -ne 0 ]; then
        # If squeue fails, assume jobs are done (safer than crashing)
        RUNNING_JOBS=0
    fi
    if [ "$RUNNING_JOBS" -eq 0 ]; then
        break
    fi
    
    CURRENT_TIME=$(date +%s)
    
    # Aggregate progress from all log files
    TOTAL_PROCESSED=0
    JOBS_COMPLETE=0
    JOBS_WRITING=0
    JOBS_RUNNING=0
    JOBS_PENDING=0
    JOBS_WITH_LOGS=0
    
    # Get job statuses from squeue
    JOB_STATUSES=$(squeue -h -j "$JID_GEN_TRAJ" -o "%i %T" 2>/dev/null || echo "")
    
    # Process each job's log file and track completion status
    # Reset completion status for this iteration
    for job_id in $(seq 1 ${NUM_ARRAY_JOBS_FINAL}); do
        JOB_COMPLETE_STATUS[$job_id]=0
    done
    
    # Disable set -e for the entire log parsing section to prevent silent exits
    set +e
    for job_id in $(seq 1 ${NUM_ARRAY_JOBS_FINAL}); do
        log_file="${SLURM_LOG_FOLDER}/gen_traj_${JID_GEN_TRAJ}_${job_id}.out"
        
        if [ -f "$log_file" ]; then
            JOBS_WITH_LOGS=$((JOBS_WITH_LOGS + 1))
            
            # Extract latest processed count from progress line
            latest_progress=$(grep "\[Progress\]" "$log_file" 2>/dev/null | tail -1 || true)
            if [ -n "$latest_progress" ]; then
                # Primary method: Extract "Cells: X/Y" and use X (completed cells for this job)
                processed=$(echo "$latest_progress" | sed -n 's/.*Cells: \([0-9]*\)\/[0-9]*.*/\1/p' 2>/dev/null || echo "")
                
                # Fallback: Extract "Processed: X" if Cells pattern didn't work
                if [ -z "$processed" ] || [ "$processed" = "0" ]; then
                    processed=$(echo "$latest_progress" | sed -n 's/.*Processed: \([0-9]*\).*/\1/p' 2>/dev/null || echo "")
                fi
                
                # Final fallback: try grep method (wrap in subshell to handle pipe failures)
                if [ -z "$processed" ] || [ "$processed" = "0" ]; then
                    processed=$(echo "$latest_progress" | grep -o "Processed: [0-9]*" 2>/dev/null | grep -o "[0-9]*" 2>/dev/null || echo "")
                fi
                
                if [ -n "$processed" ] && [ "$processed" -gt 0 ] 2>/dev/null; then
                    # Sanity check: processed should never exceed total grid cells
                    if [ "$processed" -le "$TOTAL_GRID_CELLS" ] 2>/dev/null; then
                        TOTAL_PROCESSED=$((TOTAL_PROCESSED + processed))
                    fi
                fi
            fi
            
            # Check if computation is complete
            grep -q "Trajectory Generation Complete" "$log_file" 2>/dev/null
            if [ $? -eq 0 ]; then
                JOB_COMPLETE_STATUS[$job_id]=1
            fi
        fi
    done
    set -e
    
    # Count job statuses from squeue, categorizing properly
    # Use set +e for entire parsing block to prevent script exit on parse errors
    set +e
    if [ -n "$JOB_STATUSES" ]; then
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                job_array_id=$(echo "$line" | awk '{print $1}' | grep -o "[0-9]*$" 2>/dev/null || echo "")
                status=$(echo "$line" | awk '{print $2}' 2>/dev/null || echo "")
                
                if [ -n "$job_array_id" ] && [ -n "$status" ]; then
                    is_complete=${JOB_COMPLETE_STATUS[$job_array_id]:-0}
                    
                    if [ "$is_complete" = "1" ]; then
                        # Job has completed computation
                        if [ "$status" = "RUNNING" ]; then
                            # Still in queue, so it's writing/cleanup
                            JOBS_WRITING=$((JOBS_WRITING + 1))
                        fi
                        # Note: If status is not RUNNING, we'll count it as COMPLETE below
                    else
                        # Job hasn't completed computation yet
                        if [ "$status" = "RUNNING" ]; then
                            JOBS_RUNNING=$((JOBS_RUNNING + 1))
                        elif [ "$status" = "PENDING" ]; then
                            JOBS_PENDING=$((JOBS_PENDING + 1))
                        fi
                    fi
                fi
            fi
        done <<< "$JOB_STATUSES"
    fi
    set -e
    
    # Count jobs that completed but are not in queue (fully done)
    # These are jobs that have "Complete" in log but are no longer in squeue
    # Use set +e for entire block to prevent script exit on grep failures
    set +e
    for job_id in $(seq 1 ${NUM_ARRAY_JOBS_FINAL}); do
        if [ "${JOB_COMPLETE_STATUS[$job_id]:-0}" = "1" ]; then
            # Check if this job is NOT in the queue
            echo "$JOB_STATUSES" | grep -q "${JID_GEN_TRAJ}_${job_id}" 2>/dev/null
            GREP_EXIT=$?
            if [ "$GREP_EXIT" -ne 0 ]; then
                # Not in queue, so it's fully complete
                JOBS_COMPLETE=$((JOBS_COMPLETE + 1))
            fi
        fi
    done
    set -e
    
    # Calculate percentages and ETA (with error handling for bc)
    if [ "$TOTAL_GRID_CELLS" -gt 0 ] && [ "$TOTAL_PROCESSED" -gt 0 ]; then
        CELLS_PERCENT=$(echo "scale=1; $TOTAL_PROCESSED * 100 / $TOTAL_GRID_CELLS" 2>/dev/null | bc 2>/dev/null || echo "0")
        # Validate result (use set +e to handle grep failures gracefully)
        set +e
        echo "$CELLS_PERCENT" | grep -qE '^[0-9]+\.?[0-9]*$' 2>/dev/null
        GREP_EXIT=$?
        set -e
        if [ "$GREP_EXIT" -ne 0 ]; then
            CELLS_PERCENT="0"
        fi
    else
        CELLS_PERCENT=0
    fi
    
    if [ "$NUM_ARRAY_JOBS_FINAL" -gt 0 ] && [ "$JOBS_COMPLETE" -ge 0 ]; then
        JOBS_PERCENT=$(echo "scale=1; $JOBS_COMPLETE * 100 / $NUM_ARRAY_JOBS_FINAL" 2>/dev/null | bc 2>/dev/null || echo "0")
        # Validate result (use set +e to handle grep failures gracefully)
        set +e
        echo "$JOBS_PERCENT" | grep -qE '^[0-9]+\.?[0-9]*$' 2>/dev/null
        GREP_EXIT=$?
        set -e
        if [ "$GREP_EXIT" -ne 0 ]; then
            JOBS_PERCENT="0"
        fi
    else
        JOBS_PERCENT=0
    fi
    
    # Calculate rate and ETA
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ "$ELAPSED" -gt 0 ] && [ "$TOTAL_PROCESSED" -gt "$LAST_TOTAL_PROCESSED" ]; then
        DELTA_PROCESSED=$((TOTAL_PROCESSED - LAST_TOTAL_PROCESSED))
        DELTA_TIME=$((CURRENT_TIME - LAST_PROGRESS_TIME))
        if [ "$DELTA_TIME" -gt 0 ]; then
            RATE=$(echo "scale=1; $DELTA_PROCESSED / $DELTA_TIME" | bc 2>/dev/null || echo "0")
            REMAINING=$((TOTAL_GRID_CELLS - TOTAL_PROCESSED))
            RATE_CHECK=$(echo "$RATE > 0" 2>/dev/null | bc 2>/dev/null || echo "0")
            if [ "$RATE_CHECK" = "1" ]; then
                ETA_SECONDS=$(echo "scale=0; $REMAINING / $RATE" 2>/dev/null | bc 2>/dev/null || echo "0")
                # Validate ETA_SECONDS is a number (use set +e to handle grep failures gracefully)
                set +e
                echo "$ETA_SECONDS" | grep -qE '^[0-9]+$' 2>/dev/null
                GREP_EXIT=$?
                set -e
                if [ "$GREP_EXIT" -ne 0 ]; then
                    ETA_SECONDS=0
                fi
                ETA_HOURS=$((ETA_SECONDS / 3600))
                ETA_MINS=$(((ETA_SECONDS % 3600) / 60))
                ETA_SECS=$((ETA_SECONDS % 60))
                ETA_STR=$(printf "%02d:%02d:%02d" $ETA_HOURS $ETA_MINS $ETA_SECS)
            else
                ETA_STR="--:--:--"
            fi
        else
            RATE=0
            ETA_STR="--:--:--"
        fi
    else
        RATE=0
        ETA_STR="--:--:--"
    fi
    
    # Format elapsed time
    ELAPSED_HOURS=$((ELAPSED / 3600))
    ELAPSED_MINS=$(((ELAPSED % 3600) / 60))
    ELAPSED_SECS=$((ELAPSED % 60))
    ELAPSED_STR=$(printf "%02d:%02d:%02d" $ELAPSED_HOURS $ELAPSED_MINS $ELAPSED_SECS)
    
    # Display progress (update every 10 seconds OR on first iteration)
    TIME_SINCE_LAST=$((CURRENT_TIME - LAST_PROGRESS_TIME))
    if [ "$FIRST_ITERATION" -eq 1 ] || [ "$TIME_SINCE_LAST" -ge 10 ]; then
        # Format numbers with thousands separators (compatible with different printf versions)
        TOTAL_PROCESSED_FMT=$(echo "$TOTAL_PROCESSED" | awk '{printf "%'\''d", $1}')
        TOTAL_GRID_CELLS_FMT=$(echo "$TOTAL_GRID_CELLS" | awk '{printf "%'\''d", $1}')
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [Overall Progress]"
        if [ "$TOTAL_PROCESSED" -gt 0 ]; then
            echo "    Cells: ${TOTAL_PROCESSED_FMT} / ${TOTAL_GRID_CELLS_FMT} (${CELLS_PERCENT}%) | Rate: ${RATE} cells/s | ETA: ${ETA_STR}"
        else
            echo "    Cells: ${TOTAL_PROCESSED_FMT} / ${TOTAL_GRID_CELLS_FMT} (${CELLS_PERCENT}%) | Waiting for jobs to start..."
        fi
        echo "    Jobs:  ${JOBS_COMPLETE} / ${NUM_ARRAY_JOBS_FINAL} complete (${JOBS_PERCENT}%)"
        echo ""
        echo "  [Job Status]"
        echo "    Running (computing):  ${JOBS_RUNNING} jobs"
        echo "    Writing/Cleanup:      ${JOBS_WRITING} jobs"
        echo "    Pending (queued):     ${JOBS_PENDING} jobs"
        echo "    Completed:            ${JOBS_COMPLETE} jobs"
        if [ "$JOBS_WITH_LOGS" -lt "$NUM_ARRAY_JOBS_FINAL" ]; then
            echo "    Logs available:       ${JOBS_WITH_LOGS} / ${NUM_ARRAY_JOBS_FINAL} jobs"
        fi
        echo ""
        echo "  [Time] Elapsed: ${ELAPSED_STR}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        LAST_PROGRESS_TIME=$CURRENT_TIME
        LAST_TOTAL_PROCESSED=$TOTAL_PROCESSED
        FIRST_ITERATION=0
        
        # Log memory usage every 10 iterations (helps diagnose OOM)
        ITERATION_COUNT=$((${ITERATION_COUNT:-0} + 1))
        if [ $((ITERATION_COUNT % 10)) -eq 0 ]; then
            log_memory_usage
        fi
    fi
    
    sleep 10
done

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [Final Summary]"
FINAL_TOTAL=0
# Disable set -e for log parsing to prevent silent exits
set +e
for job_id in $(seq 1 ${NUM_ARRAY_JOBS_FINAL}); do
    log_file="${SLURM_LOG_FOLDER}/gen_traj_${JID_GEN_TRAJ}_${job_id}.out"
    if [ -f "$log_file" ]; then
        latest_progress=$(grep "\[Progress\]" "$log_file" 2>/dev/null | tail -1 || true)
        if [ -n "$latest_progress" ]; then
            # Use same parsing logic as main loop
            processed=$(echo "$latest_progress" | sed -n 's/.*Cells: \([0-9]*\)\/[0-9]*.*/\1/p' 2>/dev/null || echo "")
            if [ -z "$processed" ] || [ "$processed" = "0" ]; then
                processed=$(echo "$latest_progress" | sed -n 's/.*Processed: \([0-9]*\).*/\1/p' 2>/dev/null || echo "")
            fi
            if [ -z "$processed" ] || [ "$processed" = "0" ]; then
                processed=$(echo "$latest_progress" | grep -o "Processed: [0-9]*" 2>/dev/null | grep -o "[0-9]*" 2>/dev/null || echo "0")
            fi
            if [ -n "$processed" ] && [ "$processed" -gt 0 ] 2>/dev/null; then
                # Sanity check: processed should never exceed total grid cells
                if [ "$processed" -le "$TOTAL_GRID_CELLS" ] 2>/dev/null; then
                    FINAL_TOTAL=$((FINAL_TOTAL + processed))
                fi
            fi
        fi
    fi
done
set -e
if [ "$TOTAL_GRID_CELLS" -gt 0 ] && [ "$FINAL_TOTAL" -ge 0 ]; then
    FINAL_PERCENT=$(echo "scale=1; $FINAL_TOTAL * 100 / $TOTAL_GRID_CELLS" 2>/dev/null | bc 2>/dev/null || echo "0")
    # Validate result (use set +e to handle grep failures gracefully)
    set +e
    echo "$FINAL_PERCENT" | grep -qE '^[0-9]+\.?[0-9]*$' 2>/dev/null
    GREP_EXIT=$?
    set -e
    if [ "$GREP_EXIT" -ne 0 ]; then
        FINAL_PERCENT="0"
    fi
else
    FINAL_PERCENT=0
fi
FINAL_TOTAL_FMT=$(echo "$FINAL_TOTAL" | awk '{printf "%'\''d", $1}')
TOTAL_GRID_CELLS_FMT=$(echo "$TOTAL_GRID_CELLS" | awk '{printf "%'\''d", $1}')
echo "    Total cells processed: ${FINAL_TOTAL_FMT} / ${TOTAL_GRID_CELLS_FMT} (${FINAL_PERCENT}%)"
echo "    All ${NUM_ARRAY_JOBS_FINAL} jobs completed successfully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo ">>> STAGE 1 COMPLETE."
echo "-----------------------------------------------------------------------"

# ==============================================================================
# STAGE 2A: Snapshot Metrics Calculation (FTLE, STOD, FINSTOD)
# ==============================================================================
echo ""
echo ">>> STAGE 2A: Calculating Snapshot Metrics (FTLE, STOD, FINSTOD)..."

# Calculate expected number of result files for snapshot metrics only
eval $(python3 -c "
import yaml
import math
cfg=yaml.safe_load(open('${SYSTEM_CONFIG_FILE}'))
snaps = len(cfg['time_params']['snapshot_times'])
grid_y = cfg['grid_params']['grid_resolution_y']
max_jobs = cfg['hpc_params']['max_desired_jobs']
ac = cfg.get('analysis_controls', {})

# Calculate actual number of workers that will have rows
chunk_size = (grid_y + max_jobs - 1) // max_jobs
actual_workers = min(max_jobs, (grid_y + chunk_size - 1) // chunk_size)

has_local = (ac.get('compute_ftle_forward', False) or ac.get('compute_ftle_backward', False) or
             ac.get('compute_local_stod_forward', False) or ac.get('compute_local_stod_backward', False) or
             ac.get('compute_local_finstod_forward', False) or ac.get('compute_local_finstod_backward', False) or
             ac.get('compute_fli_forward', False) or ac.get('compute_fli_backward', False) or
             ac.get('compute_ld_forward', False) or ac.get('compute_ld_backward', False))
has_global = (ac.get('compute_global_stod_forward', False) or ac.get('compute_global_stod_backward', False) or
              ac.get('compute_global_finstod_forward', False) or ac.get('compute_global_finstod_backward', False))

# Each worker produces one file per snapshot for local metrics, one per snapshot for global metrics
# Note: LB/OLB files are NOT included here (they're in Stage 2B)
total_files = 0
if has_local:
    total_files += actual_workers * snaps
if has_global:
    total_files += actual_workers * snaps

print(f'export TOTAL_EXPECTED_FILES_SNAPS={total_files}')
print(f'export ACTUAL_WORKERS={actual_workers}')
")

JID_WORK_SNAPS=$(sbatch --parsable \
       --export=ALL \
       --array=1-${NUM_ARRAY_JOBS_FINAL} \
       --output=${SLURM_LOG_FOLDER}/worker_snaps_%A_%a.out \
       --error=${SLURM_LOG_FOLDER}/worker_snaps_%A_%a.err \
       --cpus-per-task=${WORKER_CORES} \
       pipeline_core/sbatch_worker.slurm)

echo "    Submitted Job ID: $JID_WORK_SNAPS"
echo "    Actual Workers with Rows: ${ACTUAL_WORKERS}"
echo "    Expected Total Snapshot Files: ${TOTAL_EXPECTED_FILES_SNAPS}"

# Monitor job progress and wait for output files
RESULTS_FOLDER="${OUTPUT_BASE_FOLDER}/partial_results"
# Count both local and global files
if ! python3 -u pipeline_core/watch_stage.py \
    --job_id "$JID_WORK_SNAPS" \
    --pattern "${RESULTS_FOLDER}/results_chunk_*_snap_*.h5 ${RESULTS_FOLDER}/results_chunk_*_global_snap_*.h5" \
    --total "${TOTAL_EXPECTED_FILES_SNAPS}" \
    --name "Snapshot Metrics"; then
    echo "WARNING: watch_stage.py failed for Stage 2A. Continuing anyway..."
    echo "  Check if files were generated manually: ${RESULTS_FOLDER}/results_chunk_*_snap_*.h5"
fi

# Wait for Stage 2A jobs to actually finish (not just files generated)
# This is critical to avoid hitting job submission limits when submitting Stage 2B
echo "Waiting for Stage 2A jobs to complete..."
set +e
while squeue -h -j "$JID_WORK_SNAPS" 2>/dev/null | grep -q "$JID_WORK_SNAPS"; do
    sleep 5
done
set -e
echo "  -> Stage 2A jobs finished."

# Aggregate and display timing statistics
echo ""
echo "Aggregating timing statistics from workers..."
python3 -u pipeline_core/aggregate_timing.py \
    --log_folder "${SLURM_LOG_FOLDER}" \
    --job_prefix "worker_snaps" 2>/dev/null || echo "  (timing aggregation skipped - no data available)"

echo ">>> STAGE 2A COMPLETE."
echo "-----------------------------------------------------------------------"

# ==============================================================================
# STAGE 2B: LB/OLB Calculation (Time-Independent Integrals)
# ==============================================================================
# These are computed separately to improve load balancing:
# - Workers that finish snapshot metrics quickly can help with LB/OLB
# - LB/OLB are heavy computations that benefit from dedicated workers
eval $(python3 -c "
import yaml
cfg=yaml.safe_load(open('${SYSTEM_CONFIG_FILE}'))
grid_y = cfg['grid_params']['grid_resolution_y']
max_jobs = cfg['hpc_params']['max_desired_jobs']
ac = cfg.get('analysis_controls', {})

chunk_size = (grid_y + max_jobs - 1) // max_jobs
actual_workers = min(max_jobs, (grid_y + chunk_size - 1) // chunk_size)

compute_lb = ac.get('compute_lb', False)
compute_olb_local_stod = ac.get('compute_olb_local_stod', False)
compute_olb_local_finstod = ac.get('compute_olb_local_finstod', False)
compute_olb_global_stod = ac.get('compute_olb_global_stod', False)
compute_olb_global_finstod = ac.get('compute_olb_global_finstod', False)

has_lb_olb = (compute_lb or compute_olb_local_stod or compute_olb_local_finstod or 
              compute_olb_global_stod or compute_olb_global_finstod)

# Count expected LB/OLB files
total_lb_olb_files = 0
if compute_lb:
    total_lb_olb_files += actual_workers
if compute_olb_local_stod:
    total_lb_olb_files += actual_workers
if compute_olb_local_finstod:
    total_lb_olb_files += actual_workers
if compute_olb_global_stod:
    total_lb_olb_files += actual_workers
if compute_olb_global_finstod:
    total_lb_olb_files += actual_workers

print(f'export HAS_LB_OLB={str(has_lb_olb).lower()}')
print(f'export TOTAL_EXPECTED_FILES_LB_OLB={total_lb_olb_files}')
")

if [ "${HAS_LB_OLB}" = "true" ]; then
    echo ""
    echo ">>> STAGE 2B: Calculating LB/OLB (Time-Independent Integrals)..."
    
    JID_WORK_LB_OLB=$(sbatch --parsable \
           --export=ALL \
           --array=1-${NUM_ARRAY_JOBS_FINAL} \
           --output=${SLURM_LOG_FOLDER}/worker_lb_olb_%A_%a.out \
           --error=${SLURM_LOG_FOLDER}/worker_lb_olb_%A_%a.err \
           --cpus-per-task=${WORKER_CORES} \
           pipeline_core/sbatch_worker_lb_olb.slurm)
    
    echo "    Submitted Job ID: $JID_WORK_LB_OLB"
    echo "    Actual Workers with Rows: ${ACTUAL_WORKERS}"
    echo "    Expected Total LB/OLB Files: ${TOTAL_EXPECTED_FILES_LB_OLB}"
    
    # Monitor job progress and wait for output files
    if ! python3 -u pipeline_core/watch_stage.py \
        --job_id "$JID_WORK_LB_OLB" \
        --pattern "${RESULTS_FOLDER}/results_chunk_*_exact_*.h5" \
        --total "${TOTAL_EXPECTED_FILES_LB_OLB}" \
        --name "LB/OLB Metrics"; then
        echo "WARNING: watch_stage.py failed for Stage 2B. Continuing anyway..."
        echo "  Check if files were generated manually: ${RESULTS_FOLDER}/results_chunk_*_exact_*.h5"
    fi
    
    echo ">>> STAGE 2B COMPLETE."
    echo "-----------------------------------------------------------------------"
else
    echo ""
    echo ">>> STAGE 2B: Skipped (LB/OLB not enabled in config)."
    echo "-----------------------------------------------------------------------"
fi

echo ">>> STAGE 2 COMPLETE (2A + 2B)."
echo "-----------------------------------------------------------------------"

# ==============================================================================
# STAGE 3: Final Aggregation (Python/H5)
# ==============================================================================
echo ""
echo ">>> STAGE 3: Aggregating Final Results..."
if ! sbatch --wait \
       --output=${SLURM_LOG_FOLDER}/agg_results_%j.out \
       --error=${SLURM_LOG_FOLDER}/agg_results_%j.err \
       pipeline_core/sbatch_aggregate_results.slurm; then
    echo "ERROR: Stage 3 (aggregation) failed. Check logs: ${SLURM_LOG_FOLDER}/agg_results_*.err"
    exit 1
fi

echo ">>> STAGE 3 COMPLETE."
echo "-----------------------------------------------------------------------"

# ==============================================================================
# STAGE 4: Field Visualization (Python/Matplotlib)
# ==============================================================================
echo ""
echo ">>> STAGE 4: Generating Field Visualizations..."

# Create visualization output directory
VIZ_FOLDER="${OUTPUT_BASE_FOLDER}/field_visualizations"
mkdir -p "${VIZ_FOLDER}"

# Get visualization parameters from config
eval $(python3 -c "
import yaml
cfg=yaml.safe_load(open('${SYSTEM_CONFIG_FILE}'))
viz_params = cfg.get('visualization_params', {})
enabled = viz_params.get('enabled', True)
density = viz_params.get('stream_density', 2.0)
resolution = viz_params.get('grid_resolution', 150)
print(f'export VIZ_ENABLED={str(enabled).lower()}')
print(f'export VIZ_DENSITY={density}')
print(f'export VIZ_RESOLUTION={resolution}')
")

if [ "${VIZ_ENABLED}" = "true" ]; then
    echo "    Generating standard and orthogonal field plots..."
    echo "    Output: ${VIZ_FOLDER}"
    echo "    Density: ${VIZ_DENSITY}, Resolution: ${VIZ_RESOLUTION}"
    
    if python3 -u pipeline_core/visualize_fields.py \
        --config "${SYSTEM_CONFIG_FILE}" \
        --output "${VIZ_FOLDER}" \
        --density "${VIZ_DENSITY}" \
        --resolution "${VIZ_RESOLUTION}"; then
        echo ">>> STAGE 4 COMPLETE."
    else
        echo "WARNING: Field visualization failed. Check the error messages above."
        echo "  You can run it manually later with:"
        echo "  python pipeline_core/visualize_fields.py --config ${SYSTEM_CONFIG_FILE} --output ${VIZ_FOLDER}"
    fi
else
    echo ">>> STAGE 4: Skipped (visualization disabled in config)."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  FIELD MODE '${CURRENT_FIELD_MODE}' COMPLETE"
echo "║  Results:        ${FINAL_FOLDER}"
echo "║  Visualizations: ${VIZ_FOLDER}"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# End of field mode loop
done

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                         PIPELINE FINISHED SUCCESSFULLY"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Field Mode(s) Processed: ${FIELD_MODES_TO_RUN}"
echo ""
if [ "${FIELD_MODE_CONFIG}" = "both" ]; then
    echo "  Standard Results:   ${OUTPUT_BASE_FOLDER_ORIG}/standard/final_results"
    echo "  Orthogonal Results: ${OUTPUT_BASE_FOLDER_ORIG}/orthogonal/final_results"
else
    echo "  Results: ${OUTPUT_BASE_FOLDER_ORIG}/final_results"
fi
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

# Mark as normal exit to prevent false warnings
SCRIPT_EXIT_NORMAL=1
exit 0

