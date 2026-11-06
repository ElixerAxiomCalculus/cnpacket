import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "cicids"
    / "final"
)

RUNS_PATH = (
    DIRECTORY
    / "tree_final_runs.csv"
)

SUMMARY_PATH = (
    DIRECTORY
    / "tree_final_summary.csv"
)


def main():
    reports = sorted(
        DIRECTORY.glob(
            "*_seed_*_report.json"
        )
    )

    if len(reports) != 10:
        raise RuntimeError(
            f"Expected 10 reports, "
            f"found {len(reports)}"
        )

    rows = []

    for path in reports:
        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        validation = report[
            "validation"
        ][
            "metrics"
        ]

        test = report[
            "test"
        ][
            "metrics"
        ]

        rows.append(
            {
                "model": report[
                    "model"
                ],
                "seed": report[
                    "seed"
                ],
                "best_iteration": report[
                    "best_iteration"
                ],
                "training_seconds": report[
                    "training_seconds"
                ],
                "model_size_bytes": report[
                    "model_size_bytes"
                ],
                "validation_macro_f1": (
                    validation[
                        "macro_f1"
                    ]
                ),
                "test_accuracy": test[
                    "accuracy"
                ],
                "test_balanced_accuracy": (
                    test[
                        "balanced_accuracy"
                    ]
                ),
                "test_macro_f1": test[
                    "macro_f1"
                ],
                "test_weighted_f1": (
                    test[
                        "weighted_f1"
                    ]
                ),
                "test_attack_precision": (
                    test[
                        "precision_attack"
                    ]
                ),
                "test_attack_recall": (
                    test[
                        "recall_attack"
                    ]
                ),
                "test_attack_f1": (
                    test[
                        "f1_attack"
                    ]
                ),
                "test_roc_auc": test[
                    "roc_auc"
                ],
                "test_average_precision": (
                    test[
                        "average_precision"
                    ]
                ),
                "test_fpr": test[
                    "false_positive_rate"
                ],
                "test_fnr": test[
                    "false_negative_rate"
                ],
                "test_microseconds_per_sample": (
                    report[
                        "test_microseconds_per_sample"
                    ]
                ),
                "test_samples_per_second": (
                    report[
                        "test_samples_per_second"
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
                "seed"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    runs.to_csv(
        RUNS_PATH,
        index=False
    )

    numeric_columns = [
        "training_seconds",
        "model_size_bytes",
        "validation_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_attack_precision",
        "test_attack_recall",
        "test_attack_f1",
        "test_roc_auc",
        "test_average_precision",
        "test_fpr",
        "test_fnr",
        "test_microseconds_per_sample",
        "test_samples_per_second"
    ]

    summary_rows = []

    for model, frame in runs.groupby(
        "model"
    ):
        row = {
            "model": model,
            "runs": len(
                frame
            )
        }

        for column in numeric_columns:
            row[
                f"{column}_mean"
            ] = frame[
                column
            ].mean()

            row[
                f"{column}_std"
            ] = frame[
                column
            ].std(
                ddof=1
            )

        if frame[
            "best_iteration"
        ].notna().any():
            row[
                "best_iteration_mean"
            ] = frame[
                "best_iteration"
            ].mean()

            row[
                "best_iteration_std"
            ] = frame[
                "best_iteration"
            ].std(
                ddof=1
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print(
        "===== INDIVIDUAL RUNS ====="
    )

    print(
        runs.to_string(
            index=False
        )
    )

    print()
    print(
        "===== FIVE-SEED SUMMARY ====="
    )

    selected = [
        "model",
        "runs",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_attack_f1_mean",
        "test_attack_f1_std",
        "test_fpr_mean",
        "test_fpr_std",
        "test_fnr_mean",
        "test_fnr_std",
        "training_seconds_mean",
        "test_microseconds_per_sample_mean",
        "model_size_bytes_mean"
    ]

    print(
        summary[
            selected
        ].to_string(
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
        "CICIDS_TREE_FINAL_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()
