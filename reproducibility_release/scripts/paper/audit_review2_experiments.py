from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

COMPLETION = (
    ROOT
    / "artifacts"
    / "paper"
    / "experimental_completion"
)

CLOSED_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
)

OPEN_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set"
)

PHASE4_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase4"
)


checks = []


def check(
    name,
    condition,
    detail
):
    checks.append(
        {
            "item": name,
            "pass": bool(
                condition
            ),
            "detail": detail
        }
    )


closed_reports = sorted(
    CLOSED_DIRECTORY.glob(
        "cnn_bilstm_seed_*_report.json"
    )
)

check(
    "closed_set_five_runs",
    len(
        closed_reports
    )
    == 5,
    f"reports={len(closed_reports)}"
)

required_test_metric_terms = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "confusion_matrix"
}

closed_metrics_ok = True

per_class_ok = True

auc_ok = True

training_time_ok = True

inference_time_ok = True

model_size_ok = True

gpu_memory_ok = True

parameter_ok = True

for path in closed_reports:
    report = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    metrics = report.get(
        "test_metrics",
        {}
    )

    missing = (
        required_test_metric_terms
        - set(
            metrics.keys()
        )
    )

    if missing:
        closed_metrics_ok = False

    if "per_class" not in metrics:
        per_class_ok = False

    metric_keys = {
        key.lower()
        for key in metrics.keys()
    }

    if not any(
        "auc" in key
        for key in metric_keys
    ):
        auc_ok = False

    if "training_seconds" not in report:
        training_time_ok = False

    if (
        "test_batch_microseconds_per_sample"
        not in report
    ):
        inference_time_ok = False

    if "model_size_bytes" not in report:
        model_size_ok = False

    if "peak_gpu_memory_mib" not in report:
        gpu_memory_ok = False

    if "parameter_count" not in report:
        parameter_ok = False

check(
    "accuracy_macro_weighted_f1",
    closed_metrics_ok,
    "present in all five USTC closed-set reports"
)

check(
    "per_class_precision_recall_f1",
    per_class_ok,
    "per_class metrics present"
)

check(
    "roc_auc",
    auc_ok,
    "AUC metric present"
)

check(
    "training_time",
    training_time_ok,
    "training_seconds present"
)

check(
    "classifier_inference_latency",
    inference_time_ok,
    "microseconds/sample present"
)

check(
    "model_size",
    model_size_ok,
    "model_size_bytes present"
)

check(
    "training_gpu_memory",
    gpu_memory_ok,
    "peak_gpu_memory_mib present"
)

check(
    "parameter_count",
    parameter_ok,
    "parameter_count present"
)

complexity_path = (
    COMPLETION
    / "model_complexity.json"
)

complexity_ok = False

complexity_detail = "missing"

if complexity_path.exists():
    complexity = json.loads(
        complexity_path.read_text(
            encoding="utf-8"
        )
    )

    scenarios = complexity.get(
        "scenarios",
        []
    )

    complexity_ok = (
        complexity[
            "architecture"
        ][
            "encoder_parameters"
        ]
        == 67008
        and complexity[
            "architecture"
        ][
            "closed_set_parameters"
        ]
        == 68298
        and any(
            row[
                "scenario"
            ]
            == "zero_shot_8_prototypes"
            and row[
                "valid_sequence_packets"
            ]
            == 5
            and row[
                "arithmetic_flops"
            ]
            > 0
            for row in scenarios
        )
    )

    complexity_detail = (
        "encoder=67008, closed=68298, "
        "FLOPs recorded"
    )

check(
    "flops_macs_complexity",
    complexity_ok,
    complexity_detail
)

confusion_path = (
    COMPLETION
    / "closed_set_confusion_mean.csv"
)

confusion_ok = False

confusion_detail = "missing"

if confusion_path.exists():
    confusion = pd.read_csv(
        confusion_path,
        index_col=0
    )

    confusion_ok = (
        confusion.shape
        == (
            10,
            10
        )
    )

    confusion_detail = (
        f"shape={confusion.shape}"
    )

check(
    "confusion_matrix",
    confusion_ok,
    confusion_detail
)

zero_path = (
    OPEN_DIRECTORY
    / "zero_shot_summary.csv"
)

few_path = (
    OPEN_DIRECTORY
    / "few_shot_summary.csv"
)

paired_path = (
    OPEN_DIRECTORY
    / "zero_shot_paired_by_target.csv"
)

check(
    "open_set_zero_shot",
    zero_path.exists(),
    str(
        zero_path.relative_to(
            ROOT
        )
    )
)

check(
    "few_shot_k1_k5_k10",
    few_path.exists(),
    str(
        few_path.relative_to(
            ROOT
        )
    )
)

check(
    "paired_open_set_rotations",
    paired_path.exists(),
    str(
        paired_path.relative_to(
            ROOT
        )
    )
)

phase4_summary_path = (
    PHASE4_DIRECTORY
    / "benchmark"
    / "phase4_benchmark_summary.json"
)

phase4_ok = False

phase4_detail = "missing"

if phase4_summary_path.exists():
    phase4 = json.loads(
        phase4_summary_path.read_text(
            encoding="utf-8"
        )
    )

    network = phase4.get(
        "network",
        []
    )

    pipeline = phase4.get(
        "pipeline",
        {}
    )

    modes = {
        row[
            "mode"
        ]
        for row in network
    }

    phase4_ok = (
        modes
        == {
            "forwarding",
            "features",
            "inference"
        }
        and "features" in pipeline
        and "inference" in pipeline
    )

    phase4_detail = (
        "forwarding/features/inference "
        "network and pipeline metrics present"
    )

check(
    "sdn_end_to_end_latency_throughput",
    phase4_ok,
    phase4_detail
)

gpu_summary_path = (
    PHASE4_DIRECTORY
    / "benchmark"
    / "gpu_memory_summary.json"
)

gpu_deployment_ok = False

gpu_deployment_detail = "missing"

if gpu_summary_path.exists():
    gpu_summary = json.loads(
        gpu_summary_path.read_text(
            encoding="utf-8"
        )
    )

    value = gpu_summary.get(
        "incremental_memory_mib_mean"
    )

    gpu_deployment_ok = (
        value is not None
        and value > 0
    )

    gpu_deployment_detail = (
        f"incremental_gpu_mib={value}"
    )

check(
    "deployment_gpu_memory",
    gpu_deployment_ok,
    gpu_deployment_detail
)

closed_manifest = (
    CLOSED_DIRECTORY
    / "closed_set_sha256.txt"
)

open_manifest = (
    OPEN_DIRECTORY
    / "open_set_sha256.txt"
)

phase4_manifest = (
    PHASE4_DIRECTORY
    / "phase4_sha256.txt"
)

check(
    "closed_set_frozen_manifest",
    closed_manifest.exists(),
    str(
        closed_manifest.relative_to(
            ROOT
        )
    )
)

check(
    "open_set_frozen_manifest",
    open_manifest.exists(),
    str(
        open_manifest.relative_to(
            ROOT
        )
    )
)

check(
    "phase4_frozen_manifest",
    phase4_manifest.exists(),
    str(
        phase4_manifest.relative_to(
            ROOT
        )
    )
)

output_path = (
    COMPLETION
    / "review2_experiment_acceptance.csv"
)

pd.DataFrame(
    checks
).to_csv(
    output_path,
    index=False
)

print(
    "===== REVIEW-2 EXPERIMENT ACCEPTANCE ====="
)

for row in checks:
    print(
        f"{'PASS' if row['pass'] else 'FAIL'} "
        f"{row['item']}: "
        f"{row['detail']}"
    )

passed = sum(
    row[
        "pass"
    ]
    for row in checks
)

total = len(
    checks
)

print()

print(
    f"PASSED={passed}"
)

print(
    f"TOTAL={total}"
)

print(
    f"ACCEPTANCE_PERCENT="
    f"{100.0 * passed / total:.2f}"
)

print(
    f"REPORT={output_path}"
)

if passed != total:
    print(
        "REVIEW2_EXPERIMENTATION=INCOMPLETE"
    )

    sys.exit(
        1
    )

print(
    "REVIEW2_EXPERIMENTATION=PASS"
)

print(
    "EXPERIMENTATION_COMPLETE=TRUE"
)

