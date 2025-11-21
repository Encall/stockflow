#!/usr/bin/env bash
set -euo pipefail

# Set default data directory
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/tmp/stockflow}"

cmd="${1:-}"

if [[ -z "$cmd" ]]; then
  echo "No command specified."
  echo "Usage: {silver|gold} [extra args...]"
  exit 1
fi

shift  # remove first arg (the subcommand), keep the rest as extra args

case "$cmd" in
  bronze)
    echo "[ENTRYPOINT] Running BRONZE with LOCAL_DATA_DIR=$LOCAL_DATA_DIR"
    exec python src/bronze.py --local_data_dir "$LOCAL_DATA_DIR" "$@"
    ;;
  silver)
    echo "[ENTRYPOINT] Running SILVER with LOCAL_DATA_DIR=$LOCAL_DATA_DIR"
    exec python src/silver.py --local_data_dir "$LOCAL_DATA_DIR" "$@"
    ;;
  gold)
    echo "[ENTRYPOINT] Running GOLD with LOCAL_DATA_DIR=$LOCAL_DATA_DIR"
    exec python src/gold.py --local_data_dir "$LOCAL_DATA_DIR"  "$@"
    ;;
  *)
    echo "Unknown command: $cmd"
    echo "Usage: {silver|gold} [extra args...]"
    exit 1
    ;;
esac
