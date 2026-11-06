#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_controller_benchmark_mode.sh forwarding|features|inference}"

case "$MODE" in
    forwarding|features|inference)
        ;;
    *)
        exit 2
        ;;
esac

ROOT="$(
    cd "$(
        dirname "${BASH_SOURCE[0]}"
    )/../.."
    pwd
)"

ARTIFACT_DIR="$ROOT/artifacts/phase4/benchmark"

mkdir -p "$ARTIFACT_DIR"

METRICS_PATH="$ARTIFACT_DIR/${MODE}_controller_metrics.jsonl"

: > "$METRICS_PATH"

if [ "$MODE" = "inference" ]; then
    curl \
        -fsS \
        --max-time 2 \
        http://127.0.0.1:8765/health \
        >/dev/null
fi

export PHASE4_MODE="$MODE"
export PHASE4_TELEMETRY_DPID=1
export PHASE4_SEQUENCE_PACKETS=5
export PHASE4_FLOW_IDLE_SECONDS=120
export PHASE4_INFERENCE_URL="http://127.0.0.1:8765/infer"
export PHASE4_METRICS_PATH="$METRICS_PATH"

exec bash \
    "$ROOT/scripts/run_controller_phase4.sh"
