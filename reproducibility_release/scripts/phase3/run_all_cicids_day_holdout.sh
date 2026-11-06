#!/usr/bin/env bash
set -euo pipefail

SEEDS=(
    20261011
    20261012
    20261013
    20261014
    20261015
)

MODELS=(
    rf
    xgb
)

RUNS=(
    0
    1
    2
    3
    4
)

ARTIFACT_DIR="artifacts/phase3/cicids/day_holdout"

mkdir -p "$ARTIFACT_DIR"

for MODEL in "${MODELS[@]}"; do
    for RUN in "${RUNS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            REPORT="${ARTIFACT_DIR}/${MODEL}_run_${RUN}_seed_${SEED}_report.json"
            LOG="${ARTIFACT_DIR}/${MODEL}_run_${RUN}_seed_${SEED}.log"
            RESOURCE="${ARTIFACT_DIR}/${MODEL}_run_${RUN}_seed_${SEED}_resources.txt"

            if [ -f "$REPORT" ]; then
                if .venv-ml/bin/python - "$REPORT" "$MODEL" "$RUN" "$SEED" <<'PY'
import json
import sys

path, model, run, seed = sys.argv[1:]

try:
    report = json.load(
        open(
            path,
            encoding="utf-8"
        )
    )

    valid = (
        report.get("experiment")
        == "cicids2017_day_holdout"
        and report.get("model")
        == model
        and int(report.get("run"))
        == int(run)
        and int(report.get("seed"))
        == int(seed)
    )

except Exception:
    valid = False

raise SystemExit(
    0 if valid else 1
)
PY
                then
                    echo "SKIP MODEL=$MODEL RUN=$RUN SEED=$SEED"
                    continue
                fi
            fi

            echo
            echo "========================================"
            echo "MODEL=$MODEL RUN=$RUN SEED=$SEED"
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
                scripts/phase3/run_cicids_day_holdout.py \
                --model "$MODEL" \
                --run "$RUN" \
                --seed "$SEED" \
                --threads 2 \
                2>&1 \
                | tee "$LOG"
        done
    done
done

echo
echo "CICIDS_DAY_HOLDOUT_BATCH=PASS"
