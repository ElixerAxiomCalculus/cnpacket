import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
)

RUNS_PATH = (
    DIRECTORY
    / "closed_set_runs.csv"
)

SUMMARY_PATH = (
    DIRECTORY
    / "closed_set_summary.csv"
)

CLASS_PATH = (
    DIRECTORY
    / "closed_set_class_summary.csv"
)


def main():
    reports = sorted(
        DIRECTORY.glob(
            "cnn_bilstm_seed_*_report.json"
        )
    )

    if len(reports) != 5:
        raise RuntimeError(
            f"Expected 5 reports, found {len(reports)}"
        )

    expected_seeds = {
        20261011,
        20261012,
        20261013,
        20261014,
        20261015
    }

    rows = []
    class_rows = []

    for path in reports:
        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if report[
            "experiment"
        ] != "ustc_tfc2016_closed_set":
            raise RuntimeError(
                f"Unexpected experiment in {path}"
            )

        if (
            int(
                report[
                    "train_rows"
                ]
            ) != 137748
            or int(
                report[
                    "validation_rows"
                ]
            ) != 45915
            or int(
                report[
                    "test_rows"
                ]
            ) != 45918
        ):
            raise RuntimeError(
                f"Split mismatch in {path}"
            )

        metrics = report[
            "test_metrics"
        ]

        rows.append(
            {
                "seed": int(
                    report[
                        "seed"
                    ]
                ),
                "best_epoch": int(
                    report[
                        "best_epoch"
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
                ),
                "training_seconds": float(
                    report[
                        "training_seconds"
                    ]
                ),
                "test_us_per_sample": float(
                    report[
                        "test_batch_microseconds_per_sample"
                    ]
                ),
                "peak_gpu_memory_mib": float(
                    report[
                        "peak_gpu_memory_mib"
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
                )
            }
        )

        for (
            class_name,
            values
        ) in metrics[
            "per_class"
        ].items():
            class_rows.append(
                {
                    "seed": int(
                        report[
                            "seed"
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
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

    observed_seeds = set(
        runs[
            "seed"
        ]
    )

    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"Seed mismatch: {observed_seeds}"
        )

    runs.to_csv(
        RUNS_PATH,
        index=False
    )

    metrics = [
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
        "best_epoch",
        "parameter_count",
        "model_size_bytes"
    ]

    summary = {
        "model": "cnn_bilstm",
        "runs": len(
            runs
        )
    }

    for column in metrics:
        summary[
            f"{column}_mean"
        ] = float(
            runs[
                column
            ].mean()
        )

        summary[
            f"{column}_std"
        ] = float(
            runs[
                column
            ].std(
                ddof=1
            )
        )

    summary_frame = pd.DataFrame(
        [
            summary
        ]
    )

    summary_frame.to_csv(
        SUMMARY_PATH,
        index=False
    )

    classes = pd.DataFrame(
        class_rows
    )

    class_summary = (
        classes.groupby(
            "class"
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
            ),
            support=(
                "support",
                "first"
            )
        )
        .reset_index()
        .sort_values(
            "f1_mean",
            ascending=False
        )
    )

    class_summary.to_csv(
        CLASS_PATH,
        index=False
    )

    print(
        "===== USTC CLOSED-SET RUNS ====="
    )

    print(
        runs[
            [
                "seed",
                "best_epoch",
                "validation_macro_f1",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_accuracy",
                "test_roc_auc_ovr_macro",
                "training_seconds",
                "test_us_per_sample"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== FIVE-SEED SUMMARY ====="
    )

    display = [
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_weighted_f1",
        "test_roc_auc_ovr_macro",
        "test_average_precision_macro",
        "training_seconds",
        "test_us_per_sample",
        "peak_gpu_memory_mib"
    ]

    for column in display:
        print(
            f"{column.upper()}="
            f"{summary[column + '_mean']:.8f}"
            f" +/- "
            f"{summary[column + '_std']:.8f}"
        )

    print()
    print(
        "===== PER-FAMILY SUMMARY ====="
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
        f"CLASS_SUMMARY={CLASS_PATH}"
    )

    print(
        "USTC_CLOSED_SET_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()
