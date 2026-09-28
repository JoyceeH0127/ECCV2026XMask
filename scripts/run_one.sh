#!/usr/bin/env bash
# Usage: ./scripts/run_one.sh configs/sha_5.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="${1:?config yaml required}"
python train.py --config "$CFG" "${@:2}"
