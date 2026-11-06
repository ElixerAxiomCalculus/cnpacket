#!/usr/bin/env bash
set -euo pipefail

METHODS=(
    ce_msp
    supcon_proto
)

RUNS=(
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
)

ARTIFACT_DIR="artifacts/phase3/ustc/open_set"

mkdir -p "$ARTIFACT_DIR"

for METHOD in "${METHODS[@]}"; do
    for RUN in "${RUNS[@]}"; do
        REPORT="${ARTIFACT_DIR}/${METHOD}_run_${RUN}_report.json"
        LOG="${ARTIFACT_DIR}/${METHOD}_run_${RUN}.log"
        RESOURCE="${ARTIFACT_DIR}/${METHOD}_run_${RUN}_resources.txt"

        if [ -f "$REPORT" ]; then
            if .venv-ml/bin/python - "$REPORT" "$METHOD" "$RUN" <<'PY'
import json
import sys

path, method, run = sys.argv[1:]

try:
    report = json.load(
        open(
            path,
            encoding="utf-8"
        )
    )

    valid = (
        report.get("experiment")
        == "ustc_tfc2016_open_set"
        and report.get("method")
        == method
        and int(report.get("run"))
        == int(run)
        and isinstance(
            report.get("zero_shot"),
            dict
        )
    )

    if method == "supcon_proto":
        valid = (
            valid
            and isinstance(
                report.get("few_shot"),
                dict
            )
            and set(
                report[
                    "few_shot"
                ].keys()
            )
            == {
                "1",
                "5",
                "10"
            }
        )

except Exception:
    valid = False

raise SystemExit(
    0 if valid else 1
)
PY
            then
                echo "SKIP METHOD=$METHOD RUN=$RUN"
                continue
            fi
        fi

        echo
        echo "========================================"
        echo "METHOD=$METHOD RUN=$RUN"
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
            scripts/phase3/run_ustc_open_set.py \
            --method "$METHOD" \
            --run "$RUN" \
            2>&1 \
            | tee "$LOG"
    done
done

echo
echo "USTC_OPEN_SET_BATCH=PASS"
