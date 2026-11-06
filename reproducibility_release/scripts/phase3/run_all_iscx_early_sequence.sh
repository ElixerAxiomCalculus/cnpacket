#!/usr/bin/env bash
set -euo pipefail

LENGTHS=(
    5
    10
    20
    50
)

RUNS=(
    0
    1
    2
    3
    4
)

ARTIFACT_DIR="artifacts/phase3/iscx/early_sequence"

mkdir -p "$ARTIFACT_DIR"

for N in "${LENGTHS[@]}"; do
    for RUN in "${RUNS[@]}"; do
        REPORT="${ARTIFACT_DIR}/cnn_bilstm_n${N}_run_${RUN}_report.json"
        LOG="${ARTIFACT_DIR}/cnn_bilstm_n${N}_run_${RUN}.log"
        RESOURCE="${ARTIFACT_DIR}/cnn_bilstm_n${N}_run_${RUN}_resources.txt"

        if [ -f "$REPORT" ]; then
            if .venv-ml/bin/python - "$REPORT" "$N" "$RUN" <<'PY'
import json
import sys

path, sequence_length, run = sys.argv[1:]

try:
    report = json.load(
        open(
            path,
            encoding="utf-8"
        )
    )

    valid = (
        report.get("experiment")
        == "iscxvpn2016_early_classification"
        and report.get("model")
        == "cnn_bilstm"
        and int(report.get("sequence_length"))
        == int(sequence_length)
        and int(report.get("run"))
        == int(run)
        and int(report.get("cohort_flows"))
        == 1811
    )

except Exception:
    valid = False

raise SystemExit(
    0 if valid else 1
)
PY
            then
                echo "SKIP N=$N RUN=$RUN"
                continue
            fi
        fi

        echo
        echo "========================================"
        echo "N=$N RUN=$RUN"
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
            scripts/phase3/run_iscx_early_sequence.py \
            --sequence-length "$N" \
            --run "$RUN" \
            2>&1 \
            | tee "$LOG"
    done
done

echo
echo "ISCX_EARLY_SEQUENCE_BATCH=PASS"
