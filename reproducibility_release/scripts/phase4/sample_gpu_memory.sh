#!/usr/bin/env bash
set -euo pipefail

PID="${1:?usage: sample_gpu_memory.sh PID}"
OUTPUT="${2:?usage: sample_gpu_memory.sh PID OUTPUT}"

: > "$OUTPUT"

for _ in $(seq 1 100)
do
    VALUE="$(
        nvidia-smi \
            --query-compute-apps=pid,used_memory \
            --format=csv,noheader,nounits \
        | awk -F',' \
            -v target="$PID" \
            '$1 + 0 == target {
                gsub(/ /, "", $2)
                print $2
            }'
    )"

    if [ -n "$VALUE" ]; then
        printf '%s\n' "$VALUE" >> "$OUTPUT"
    fi

    sleep 0.1
done
