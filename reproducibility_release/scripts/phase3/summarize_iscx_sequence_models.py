import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "iscx"
    / "sequence_models"
)

RUNS_PATH = (
    DIRECTORY
    / "sequence_model_runs.csv"
)

SUMMARY_PATH = (
    DIRECTORY
    / "sequence_model_summary.csv"
)

DOMAIN_PATH = (
    DIRECTORY
    / "sequence_model_domain_summary.csv"
)

CLASS_PATH = (
    DIRECTORY
    / "sequence_model_class_summary.csv"
)


def main():
    reports = sorted(
        DIRECTORY.glob(
            "*_run_*_report.json"
        )
    )

    if len(reports) != 20:
        raise RuntimeError(
            f"Expected 20 reports, "
            f"found {len(reports)}"
        )

    rows = []
    domain_rows = []
    class_rows = []

    for path in reports:
        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        metrics = report[
            "test_metrics"
        ]

        rows.append(
            {
                "model": report[
                    "model"
                ],
                "run": int(
                    report[
                        "run"
                    ]
                ),
                "test_fold": int(
                    report[
                        "test_fold"
                    ]
                ),
                "best_epoch": int(
                    report[
                        "best_epoch"
                    ]
                ),
                "epochs_completed": int(
                    report[
                        "epochs_completed"
                    ]
                ),
                "parameter_count": int(
                    report[
                        "parameter_count"
                    ]
                ),
                "model_size_bytes": int(
                    report[
                        "model_size_bytes"
                    ]
                ),
                "training_seconds": float(
                    report[
                        "training_seconds"
                    ]
                ),
                "peak_gpu_memory_mib": float(
                    report[
                        "peak_gpu_memory_mib"
                    ]
                ),
                "test_us_per_sample": float(
                    report[
                        "test_batch_microseconds_per_sample"
                    ]
                ),
                "validation_macro_f1": float(
                    report[
                        "validation_metrics"
                    ][
                        "macro_f1"
                    ]
                ),
                "test_accuracy": float(
                    metrics[
                        "accuracy"
                    ]
                ),
                "test_balanced_accuracy": float(
                    metrics[
                        "balanced_accuracy"
                    ]
                ),
                "test_macro_f1": float(
                    metrics[
                        "macro_f1"
                    ]
                ),
                "test_weighted_f1": float(
                    metrics[
                        "weighted_f1"
                    ]
                ),
                "test_roc_auc_ovr_macro": float(
                    metrics[
                        "roc_auc_ovr_macro"
                    ]
                ),
                "test_average_precision_macro": float(
                    metrics[
                        "average_precision_macro"
                    ]
                )
            }
        )

        for domain in [
            "nonvpn",
            "vpn"
        ]:
            domain_metrics = report[
                "test_domain_metrics"
            ][
                domain
            ]

            domain_rows.append(
                {
                    "model": report[
                        "model"
                    ],
                    "run": int(
                        report[
                            "run"
                        ]
                    ),
                    "test_fold": int(
                        report[
                            "test_fold"
                        ]
                    ),
                    "domain": domain,
                    "samples": int(
                        domain_metrics[
                            "samples"
                        ]
                    ),
                    "accuracy": float(
                        domain_metrics[
                            "accuracy"
                        ]
                    ),
                    "balanced_accuracy": float(
                        domain_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "macro_f1": float(
                        domain_metrics[
                            "macro_f1"
                        ]
                    ),
                    "weighted_f1": float(
                        domain_metrics[
                            "weighted_f1"
                        ]
                    )
                }
            )

        for class_name, values in metrics[
            "per_class"
        ].items():
            class_rows.append(
                {
                    "model": report[
                        "model"
                    ],
                    "run": int(
                        report[
                            "run"
                        ]
                    ),
                    "test_fold": int(
                        report[
                            "test_fold"
                        ]
                    ),
                    "class": class_name,
                    "precision": float(
                        values[
                            "precision"
                        ]
                    ),
                    "recall": float(
                        values[
                            "recall"
                        ]
                    ),
                    "f1": float(
                        values[
                            "f1"
                        ]
                    ),
                    "support": int(
                        values[
                            "support"
                        ]
                    )
                }
            )

    runs = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "model",
                "run"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    counts = runs.groupby(
        "model"
    ).size()

    if not (
        counts == 5
    ).all():
        raise RuntimeError(
            f"Expected five folds per model: "
            f"{counts.to_dict()}"
        )

    runs.to_csv(
        RUNS_PATH,
        index=False
    )

    metric_columns = [
        "validation_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_roc_auc_ovr_macro",
        "test_average_precision_macro",
        "training_seconds",
        "test_us_per_sample",
        "peak_gpu_memory_mib",
        "parameter_count",
        "model_size_bytes",
        "best_epoch"
    ]

    summary_rows = []

    for model, frame in runs.groupby(
        "model"
    ):
        row = {
            "model": model,
            "folds": len(
                frame
            )
        }

        for column in metric_columns:
            row[
                f"{column}_mean"
            ] = float(
                frame[
                    column
                ].mean()
            )

            row[
                f"{column}_std"
            ] = float(
                frame[
                    column
                ].std(
                    ddof=1
                )
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "test_macro_f1_mean",
        ascending=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    domains = pd.DataFrame(
        domain_rows
    )

    domain_summary = (
        domains.groupby(
            [
                "model",
                "domain"
            ]
        )
        .agg(
            macro_f1_mean=(
                "macro_f1",
                "mean"
            ),
            macro_f1_std=(
                "macro_f1",
                "std"
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean"
            )
        )
        .reset_index()
    )

    domain_summary.to_csv(
        DOMAIN_PATH,
        index=False
    )

    classes = pd.DataFrame(
        class_rows
    )

    class_summary = (
        classes.groupby(
            [
                "model",
                "class"
            ]
        )
        .agg(
            precision_mean=(
                "precision",
                "mean"
            ),
            precision_std=(
                "precision",
                "std"
            ),
            recall_mean=(
                "recall",
                "mean"
            ),
            recall_std=(
                "recall",
                "std"
            ),
            f1_mean=(
                "f1",
                "mean"
            ),
            f1_std=(
                "f1",
                "std"
            )
        )
        .reset_index()
    )

    class_summary.to_csv(
        CLASS_PATH,
        index=False
    )

    print(
        "===== FIVE-FOLD MODEL SUMMARY ====="
    )

    display_columns = [
        "model",
        "folds",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_balanced_accuracy_mean",
        "test_roc_auc_ovr_macro_mean",
        "test_accuracy_mean",
        "training_seconds_mean",
        "test_us_per_sample_mean",
        "peak_gpu_memory_mib_mean",
        "parameter_count_mean",
        "model_size_bytes_mean"
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== DOMAIN SUMMARY ====="
    )

    print(
        domain_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PER-CLASS SUMMARY ====="
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"RUNS={RUNS_PATH}"
    )

    print(
        f"SUMMARY={SUMMARY_PATH}"
    )

    print(
        f"DOMAIN_SUMMARY={DOMAIN_PATH}"
    )

    print(
        f"CLASS_SUMMARY={CLASS_PATH}"
    )

    print(
        "ISCX_SEQUENCE_MODEL_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()
