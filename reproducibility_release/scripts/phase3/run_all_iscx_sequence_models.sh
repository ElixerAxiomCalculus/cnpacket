#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    cnn
    lstm
    bilstm
    cnn_bilstm
)

RUNS=(
    0
    1
    2
    3
    4
)

ARTIFACT_DIR="artifacts/phase3/iscx/sequence_models"

mkdir -p "$ARTIFACT_DIR"

for MODEL in "${MODELS[@]}"; do
    for RUN in "${RUNS[@]}"; do
        REPORT="${ARTIFACT_DIR}/${MODEL}_run_${RUN}_report.json"
        LOG="${ARTIFACT_DIR}/${MODEL}_run_${RUN}.log"
        RESOURCE="${ARTIFACT_DIR}/${MODEL}_run_${RUN}_resources.txt"

        if [ -f "$REPORT" ]; then
            if .venv-ml/bin/python - "$REPORT" "$MODEL" "$RUN" <<'PY'
import json
import sys

path, model, run = sys.argv[1:]

try:
    report = json.load(
        open(
            path,
            encoding="utf-8"
        )
    )

    valid = (
        report.get("experiment")
        == "iscxvpn2016_sequence_model_ladder"
        and report.get("model")
        == model
        and int(report.get("run"))
        == int(run)
        and int(report.get("sequence_length"))
        == 5
    )

except Exception:
    valid = False

raise SystemExit(
    0 if valid else 1
)
PY
            then
                echo "SKIP MODEL=$MODEL RUN=$RUN"
                continue
            fi
        fi

        echo
        echo "========================================"
        echo "MODEL=$MODEL RUN=$RUN"
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
            scripts/phase3/run_iscx_sequence_model.py \
            --model "$MODEL" \
            --run "$RUN" \
            2>&1 \
            | tee "$LOG"
    done
done

echo
echo "ISCX_SEQUENCE_MODEL_BATCH=PASS"
