#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <base-run-dir>" >&2
  exit 1
fi

base_dir="$1"
parent_dir="$(dirname "$base_dir")"
mkdir -p "$parent_dir"

candidate="$base_dir"
suffix=1
while ! mkdir "$candidate" 2>/dev/null; do
  candidate="${base_dir}_${suffix}"
  suffix=$((suffix + 1))
done

printf '%s\n' "$candidate"
