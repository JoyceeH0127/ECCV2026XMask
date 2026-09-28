#!/usr/bin/env bash
# Run all SHA / UCF / JHU percentage configs sequentially.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
  configs/sha_5.yaml
  configs/sha_10.yaml
  configs/sha_40.yaml
  configs/ucf_5.yaml
  configs/ucf_10.yaml
  configs/ucf_40.yaml
  configs/jhu_5.yaml
  configs/jhu_10.yaml
  configs/jhu_40.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "======== training $cfg ========"
  python train.py --config "$cfg"
done
