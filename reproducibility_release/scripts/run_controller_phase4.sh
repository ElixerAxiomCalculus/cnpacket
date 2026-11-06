#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/logs"

"$ROOT/.venv-controller/bin/ryu-manager" \
    --ofp-tcp-listen-port 6653 \
    "$ROOT/controller/phase4_monitor.py" \
    2>&1 | tee "$ROOT/logs/controller.log"
