#!/bin/bash
#SBATCH --job-name=CELL_CANARY
#SBATCH --partition=standard
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_cell_canary_%j.out
#SBATCH --error=slurm_cell_canary_%j.err

set -euo pipefail

# --- Change to submit directory ---
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "======================================================================="
echo " Cell Canary Test"
echo "  Submit dir : ${SLURM_SUBMIT_DIR:-$PWD}"
echo "  Work dir   : $(pwd)"
echo "  Job ID     : ${SLURM_JOB_ID:-N/A}"
echo "======================================================================="

# --- Validate config path ---
if [[ -z "${CFG:-}" ]]; then
  echo "ERROR: You must pass --export=CFG=<path/to/config.yaml> to sbatch" >&2
  exit 1
fi

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: Config file not found: $CFG" >&2
  exit 1
fi

# Resolve absolute path safely
SYSTEM_CONFIG_FILE="$(readlink -f "$CFG" 2>/dev/null || realpath "$CFG" 2>/dev/null || echo "$CFG")"
export SYSTEM_CONFIG_FILE

echo "Using config: $SYSTEM_CONFIG_FILE"

# --- Environment setup ---
module --force purge >/dev/null 2>&1 || true
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# Activate venv
if [[ ! -f "generic_env/bin/activate" ]]; then
  echo "ERROR: virtualenv not found at generic_env/bin/activate" >&2
  exit 1
fi
source generic_env/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# --- Run cell canary ---
echo "Starting cell_canary.py ..."
python pipeline_core/cell_canary.py
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
  echo ""
  echo "!!! cell_canary.py exited with non-zero status: $EXIT_CODE !!!"
  exit $EXIT_CODE
fi

echo ""
echo "Cell canary test finished successfully."
echo "Results saved in: {output_base_folder}/Pipeline Code Canary Cell Results/"
exit 0


