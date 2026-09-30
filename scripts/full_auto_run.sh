#!/usr/bin/env bash
set -euo pipefail

script_list=(
    "scripts/mean_s4.sh"
    "scripts/mean.sh"
    "scripts/mean_flow_s4_only.sh"
    "scripts/mean_flow_only.sh"
    "scripts/mean_flow_s4.sh"
    "scripts/mean_ablation.sh"
    "scripts/mean_ablation_sf_only.sh"
    "scripts/mean_ablation_zsf_only.sh"
)

# Check if the script exists
for script in "${script_list[@]}"; do
    if [ ! -f "$script" ]; then
        echo "Error: $script does not exist."
        exit 1
    fi
done

for script in "${script_list[@]}"; do
    echo "Running $script..."
    bash "$script"
done