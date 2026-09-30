#!/usr/bin/env bash
set -euo pipefail


# =========================================================
# Dataset configurations
#
# DATA_PATHS[i], EXP_BASES[i], CONFIG_PATHS[i]
# must correspond to the same dataset.
# =========================================================

DATA_PATHS=(
    # "/network-volume/nhphuc/dataset/endonerf/pulling_soft_tissues"
    # "/network-volume/nhphuc/dataset/endonerf/cutting_tissues_twice"
    "/network-volume/nhphuc/dataset/stereomis/p2_1_1_247"
    "/network-volume/nhphuc/dataset/stereomis/p2_6_5000_200"
    "/network-volume/nhphuc/dataset/stereomis/p3_9100_368"
    "/network-volume/nhphuc/dataset/stereomis/p3_11000_401"
)

EXP_BASES=(
    # "endonerf/pulling"
    # "endonerf/cutting"
    "stereomis/p2_1_1_247"
    "stereomis/p2_6_5000_200"
    "stereomis/p3_9100_368"
    "stereomis/p3_11000_401"
)

CONFIG_PATHS=(
    # "arguments/endonerf/default.py"
    # "arguments/endonerf/default.py"
    "arguments/stereomis/default.py"
    "arguments/stereomis/default.py"
    "arguments/stereomis/default.py"
    "arguments/stereomis/default.py"
)

NUM_RUNS=2

# =========================================================
# Run one experiment setting
# =========================================================

run_experiment() {
    DATA_PATH=$1
    CONFIG=$2
    EXP_NAME=$3

    OUT_DIR="output/mean_eval_${EXP_NAME}"
    TARGET_DIR="output/${EXP_NAME}"

    mkdir -p "$OUT_DIR"

    for i in $(seq -w 1 "$NUM_RUNS"); do

        RUN_OUT_DIR="${OUT_DIR}/run${i}"
        mkdir -p "$RUN_OUT_DIR"

        # Clean previous training output
        if [ -d "$TARGET_DIR" ]; then
            echo "Removing existing directory: $TARGET_DIR"
            rm -rf "$TARGET_DIR"
        fi

        echo
        echo "=================================================="
        echo "Dataset    : $DATA_PATH"
        echo "Config     : $CONFIG"
        echo "Experiment : $EXP_NAME"
        echo "Run        : $i/$NUM_RUNS"
        echo "=================================================="

        python train_flow.py \
            -s "$DATA_PATH" \
            --expname "$EXP_NAME" \
            --configs "$CONFIG" \
            --alpha_3d 0.0

        python render.py \
            --model_path "$TARGET_DIR" \
            --skip_train \
            --configs "$CONFIG"
            # --skip_video \

        python metrics.py \
            --model_path "$TARGET_DIR" \
            -p test

        cp \
            "$TARGET_DIR/per_view.json" \
            "$RUN_OUT_DIR/per_view.json"

        cp \
            "$TARGET_DIR/results.json" \
            "$RUN_OUT_DIR/results.json"
    done
}


# =========================================================
# Verify dataset lists
# =========================================================

if [ "${#DATA_PATHS[@]}" -ne "${#EXP_BASES[@]}" ] || \
   [ "${#DATA_PATHS[@]}" -ne "${#CONFIG_PATHS[@]}" ]; then
    echo "ERROR: DATA_PATHS, EXP_BASES, and CONFIG_PATHS must have the same length."
    exit 1
fi


# =========================================================
# Run all datasets
# =========================================================

for idx in "${!DATA_PATHS[@]}"; do

    DATA_PATH="${DATA_PATHS[$idx]}"
    EXP_BASE="${EXP_BASES[$idx]}"
    CONFIG="${CONFIG_PATHS[$idx]}"

    echo
    echo "##################################################"
    echo "Dataset $((idx + 1))/${#DATA_PATHS[@]}"
    echo "DATA_PATH = $DATA_PATH"
    echo "EXP_BASE  = $EXP_BASE"
    echo "CONFIG    = $CONFIG"
    echo "##################################################"


    # =====================================================
    # alpha_3d ablation
    #
    # flow_loss_weight is left at its default value.
    # =====================================================

    EXP_NAME="${EXP_BASE}_flow_only"

    run_experiment \
        "$DATA_PATH" \
        "$CONFIG" \
        "$EXP_NAME"

done