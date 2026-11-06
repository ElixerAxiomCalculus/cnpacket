#!/usr/bin/env bash
set -euo pipefail

SEEDS=(
    20261011
    20261012
    20261013
    20261014
    20261015
)

ARTIFACT_DIR="artifacts/phase3/ustc/closed_set"

mkdir -p "$ARTIFACT_DIR"

for SEED in "${SEEDS[@]}"; do
    REPORT="${ARTIFACT_DIR}/cnn_bilstm_seed_${SEED}_report.json"
    LOG="${ARTIFACT_DIR}/cnn_bilstm_seed_${SEED}.log"
    RESOURCE="${ARTIFACT_DIR}/cnn_bilstm_seed_${SEED}_resources.txt"

    if [ -f "$REPORT" ]; then
        if .venv-ml/bin/python - "$REPORT" "$SEED" <<'PY'
import json
import sys

path, seed = sys.argv[1:]

try:
    report = json.load(
        open(
            path,
            encoding="utf-8"
        )
    )

    valid = (
        report.get("experiment")
        == "ustc_tfc2016_closed_set"
        and report.get("model")
        == "cnn_bilstm"
        and int(report.get("seed"))
        == int(seed)
        and int(report.get("train_rows"))
        == 137748
        and int(report.get("validation_rows"))
        == 45915
        and int(report.get("test_rows"))
        == 45918
    )

except Exception:
    valid = False

raise SystemExit(
    0 if valid else 1
)
PY
        then
            echo "SKIP SEED=$SEED"
            continue
        fi
    fi

    echo
    echo "========================================"
    echo "SEED=$SEED"
    echo "========================================"

    /usr/bin/time -v \
        -o "$RESOURCE" \
        env \
        PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 \
        NUMEXPR_NUM_THREADS=2 \
        .venv-ml/bin/python \
        scripts/phase3/run_ustc_closed_set.py \
        --seed "$SEED" \
        2>&1 \
        | tee "$LOG"
done

echo
echo "USTC_CLOSED_SET_BATCH=PASS"
