#!/bin/bash
#SBATCH --job-name=traj_merge
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
# ==============================================================================
#           TRAJECTORY MERGE - SLURM SCRIPT
# ==============================================================================
# This job runs after trajectory generation completes.
# It merges all partial HDF5 files into a single downloadable file.
# ==============================================================================

set -e

echo "======================================================================="
echo "   TRAJECTORY MERGE"
echo "   Started: $(date)"
echo "======================================================================="

# Environment setup
module --force purge > /dev/null 2>&1 || true
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0
module load HDF5/1.14.0-gompi-2022b

source generic_env/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Get paths from environment (set by master script)
if [ -z "$OUTPUT_BASE_FOLDER" ]; then
    echo "ERROR: OUTPUT_BASE_FOLDER not set" >&2
    exit 1
fi

if [ -z "$SYSTEM_CONFIG_FILE" ]; then
    echo "ERROR: SYSTEM_CONFIG_FILE not set" >&2
    exit 1
fi

TRAJECTORY_FOLDER="${OUTPUT_BASE_FOLDER}/trajectory_cache/partial"

# Verify trajectory files exist
COMPLETED_FILES=$(ls -1 "${TRAJECTORY_FOLDER}"/chunk_*.h5 2>/dev/null | wc -l || echo 0)
echo "Found ${COMPLETED_FILES} trajectory chunk files"

if [ "$COMPLETED_FILES" -eq 0 ]; then
    echo "ERROR: No trajectory files found in ${TRAJECTORY_FOLDER}"
    exit 1
fi

# Run merge
echo ""
echo ">>> Merging trajectory files..."
python3 pipeline_core/merge_trajectories.py "${OUTPUT_BASE_FOLDER}" --config "${SYSTEM_CONFIG_FILE}"

# Verify output
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
    echo ""
    echo "  Finished: $(date)"
    echo "======================================================================="
else
    echo "ERROR: Merge failed - output file not created"
    exit 1
fi

