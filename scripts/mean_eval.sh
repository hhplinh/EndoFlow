#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=$1
EXP_BASE=$2
CONFIG=$3

OUT_DIR="output/mean_eval_${EXP_BASE}"

TARGET_DIR="output/$EXP_BASE"

if [ -d "$TARGET_DIR" ]; then
  echo "About to remove directory: $TARGET_DIR"
  read -r -p "Are you sure? Type 'yes' to continue: " CONFIRM
  if [ "$CONFIRM" == "yes" ]; then
    rm -rf "$TARGET_DIR"
  fi
fi

mkdir -p "$OUT_DIR"

for i in $(seq -w 1 2); do
  EXP_NAME="${EXP_BASE}"
  RUN_OUT_DIR="${OUT_DIR}/run${i}"
  mkdir -p "$RUN_OUT_DIR"

  echo "===== Run $i: $EXP_NAME ====="

  # python train.py -s "$DATA_PATH" --expname "$EXP_NAME" --configs "$CONFIG"
  python train_sf.py -s "$DATA_PATH" --expname "$EXP_NAME" --configs "$CONFIG"
  # python train_zsf.py -s "$DATA_PATH" --expname "$EXP_NAME" --configs "$CONFIG"
  # python train_flow.py -s "$DATA_PATH" --expname "$EXP_NAME" --configs "$CONFIG"
  python render.py --model_path "output/$EXP_NAME"  --skip_train --skip_video --configs "$CONFIG"
  python metrics.py --model_path "output/$EXP_NAME" -p test

  cp "output/$EXP_NAME/per_view.json" "$RUN_OUT_DIR/per_view.json"
  cp "output/$EXP_NAME/results.json"  "$RUN_OUT_DIR/results.json"
done

# python train.py -s /volume/data/endonerf/pulling_soft_tissues --expname endonerf/pulling --configs arguments/endonerf/default.py
# python train.py -s /volume/data/endonerf/cutting_tissues_twice --expname endonerf/cutting --configs arguments/endonerf/default.py

# python render.py --model_path output/endonerf/cutting  --skip_train --skip_video --configs arguments/endonerf/default.py
# python render.py --model_path output/endonerf/pulling  --skip_train --skip_video --configs arguments/endonerf/default.py

# python metrics.py --model_path output/endonerf/cutting -p test
# python metrics.py --model_path output/endonerf/pulling -p test