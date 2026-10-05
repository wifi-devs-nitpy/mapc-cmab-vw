#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VW Epsilon Hyperparameter Tuning
# ============================================================
#
# Usage:
#   bash run_tuning.sh
#
# Optional:
#   bash run_tuning.sh oat
#   bash run_tuning.sh uniform
#   bash run_tuning.sh grid
#
# Environment variables can override the defaults below:
#   N_RUNS=10 N_STEPS=20000 bash run_tuning.sh
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/tune_vw_epsilon.py"

MODE="${1:-grid}"

# -----------------------------
# Experiment settings
# -----------------------------
N_RUNS="${N_RUNS:-5}"
N_STEPS="${N_STEPS:-6000}"
SEED="${SEED:-42}"
D_AP="${D_AP:-10}"
D_STA_1="${D_STA_1:-2}"
D_STA_2="${D_STA_2:-4}"
N_LINKS="${N_LINKS:-3}"
N_TX_POWER_LEVELS="${N_TX_POWER_LEVELS:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
CONFIDENCE="${CONFIDENCE:-0.99}"

OUTPUT_DIR="${OUTPUT_DIR:-./tuning_results}"

# Set FORCE=1 to ignore cached results.
FORCE="${FORCE:-0}"

# Set SHOW=1 to display plots interactively.
SHOW="${SHOW:-0}"

# -----------------------------
# Check Python script
# -----------------------------
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Could not find ${PYTHON_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo " VW Epsilon Hyperparameter Tuning"
echo "============================================================"
echo "Mode              : ${MODE}"
echo "Runs              : ${N_RUNS}"
echo "Steps             : ${N_STEPS}"
echo "Seed              : ${SEED}"
echo "AP distance       : ${D_AP}"
echo "STA distance #1   : ${D_STA_1}"
echo "STA distance #2   : ${D_STA_2}"
echo "Links             : ${N_LINKS}"
echo "TX power levels   : ${N_TX_POWER_LEVELS}"
echo "Window size       : ${WINDOW_SIZE}"
echo "Confidence        : ${CONFIDENCE}"
echo "Output directory  : ${OUTPUT_DIR}"
echo "Force rerun       : ${FORCE}"
echo "Show plots        : ${SHOW}"
echo "============================================================"
echo

# -----------------------------
# Build optional flags
# -----------------------------
EXTRA_ARGS=()

if [[ "${FORCE}" == "1" ]]; then
    EXTRA_ARGS+=(--force)
fi

if [[ "${SHOW}" == "1" ]]; then
    EXTRA_ARGS+=(--show)
fi

# -----------------------------
# Run experiment
# -----------------------------
python "${PYTHON_SCRIPT}" \
    --mode "${MODE}" \
    --n-runs "${N_RUNS}" \
    --n-steps "${N_STEPS}" \
    --seed "${SEED}" \
    --d-ap "${D_AP}" \
    --d-sta-1 "${D_STA_1}" \
    --d-sta-2 "${D_STA_2}" \
    --n-links "${N_LINKS}" \
    --n-tx-power-levels "${N_TX_POWER_LEVELS}" \
    --window-size "${WINDOW_SIZE}" \
    --confidence "${CONFIDENCE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo
echo "============================================================"
echo " Tuning completed"
echo "============================================================"
echo "Results: ${OUTPUT_DIR}"
echo
echo "Summary:"
echo "  ${OUTPUT_DIR}/summary.csv"
echo
echo "Plots:"
echo "  ${OUTPUT_DIR}/plots/"
echo
