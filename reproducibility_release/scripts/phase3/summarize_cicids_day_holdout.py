import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "cicids"
    / "day_holdout"
)

RUNS_PATH = (
    DIRECTORY
    / "day_holdout_runs.csv"
)

DAY_SUMMARY_PATH = (
    DIRECTORY
    / "day_holdout_day_summary.csv"
)

BINARY_SUMMARY_PATH = (
    DIRECTORY
    / "day_holdout_binary_summary.csv"
)


def safe_metric(
    metrics,
    key
):
    value = metrics.get(
        key
    )

    if value is None:
        return np.nan

    return float(
        value
    )


def main():
    reports = sorted(
        DIRECTORY.glob(
            "*_run_*_seed_*_report.json"
        )
    )

    if len(reports) != 50:
        raise RuntimeError(
            f"Expected 50 reports, "
            f"found {len(reports)}"
        )

    rows = []

    for path in reports:
        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        metrics = report[
            "test"
        ][
            "metrics"
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
                "seed": int(
                    report[
                        "seed"
                    ]
                ),
                "test_day": report[
                    "test_days"
                ][0],
                "evaluation_scope": report[
                    "evaluation_scope"
                ],
                "training_seconds": float(
                    report[
                        "training_seconds"
                    ]
                ),
                "best_iteration": report[
                    "best_iteration"
                ],
                "model_size_bytes": int(
                    report[
                        "model_size_bytes"
                    ]
                ),
                "test_samples": int(
                    metrics[
                        "samples"
                    ]
                ),
                "test_benign": int(
                    metrics[
                        "benign"
                    ]
                ),
                "test_attack": int(
                    metrics[
                        "attack"
                    ]
                ),
                "test_accuracy": safe_metric(
                    metrics,
                    "accuracy"
                ),
                "test_balanced_accuracy": safe_metric(
                    metrics,
                    "balanced_accuracy"
                ),
                "test_macro_f1": safe_metric(
                    metrics,
                    "macro_f1"
                ),
                "test_weighted_f1": safe_metric(
                    metrics,
                    "weighted_f1"
                ),
                "test_attack_precision": safe_metric(
                    metrics,
                    "precision_attack"
                ),
                "test_attack_recall": safe_metric(
                    metrics,
                    "recall_attack"
                ),
                "test_attack_f1": safe_metric(
                    metrics,
                    "f1_attack"
                ),
                "test_roc_auc": safe_metric(
                    metrics,
                    "roc_auc"
                ),
                "test_average_precision": safe_metric(
                    metrics,
                    "average_precision"
                ),
                "test_fpr": safe_metric(
                    metrics,
                    "false_positive_rate"
                ),
                "test_fnr": safe_metric(
                    metrics,
                    "false_negative_rate"
                ),
                "false_positives": int(
                    metrics[
                        "false_positives"
                    ]
                ),
                "false_negatives": int(
                    metrics[
                        "false_negatives"
                    ]
                ),
                "true_negatives": int(
                    metrics[
                        "true_negatives"
                    ]
                ),
                "true_positives": int(
                    metrics[
                        "true_positives"
                    ]
                ),
                "test_microseconds_per_sample": float(
                    report[
                        "test"
                    ][
                        "microseconds_per_sample"
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
                "run",
                "seed"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        runs.groupby(
            [
                "model",
                "run"
            ]
        ).size()
        != 5
    ).any():
        raise RuntimeError(
            "Every model/run must have "
            "exactly five seeds"
        )

    runs.to_csv(
        RUNS_PATH,
        index=False
    )

    metric_columns = [
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
        "training_seconds",
        "test_microseconds_per_sample",
        "model_size_bytes"
    ]

    summary_rows = []

    for (
        model,
        run,
        test_day,
        scope
    ), frame in runs.groupby(
        [
            "model",
            "run",
            "test_day",
            "evaluation_scope"
        ],
        sort=True
    ):
        row = {
            "model": model,
            "run": run,
            "test_day": test_day,
            "evaluation_scope": scope,
            "seeds": len(
                frame
            ),
            "test_samples": int(
                frame[
                    "test_samples"
                ].iloc[0]
            ),
            "test_benign": int(
                frame[
                    "test_benign"
                ].iloc[0]
            ),
            "test_attack": int(
                frame[
                    "test_attack"
                ].iloc[0]
            )
        }

        for column in metric_columns:
            values = frame[
                column
            ].dropna()

            if len(values) == 0:
                row[
                    f"{column}_mean"
                ] = np.nan

                row[
                    f"{column}_std"
                ] = np.nan
            else:
                row[
                    f"{column}_mean"
                ] = float(
                    values.mean()
                )

                row[
                    f"{column}_std"
                ] = (
                    float(
                        values.std(
                            ddof=1
                        )
                    )
                    if len(values) > 1
                    else 0.0
                )

        summary_rows.append(
            row
        )

    day_summary = pd.DataFrame(
        summary_rows
    )

    day_summary.to_csv(
        DAY_SUMMARY_PATH,
        index=False
    )

    binary = runs[
        runs[
            "test_attack"
        ]
        > 0
    ].copy()

    binary_summary_rows = []

    for model, frame in binary.groupby(
        "model"
    ):
        binary_summary_rows.append(
            {
                "model": model,
                "runs": len(
                    frame
                ),
                "days": int(
                    frame[
                        "test_day"
                    ].nunique()
                ),
                "macro_f1_mean": float(
                    frame[
                        "test_macro_f1"
                    ].mean()
                ),
                "macro_f1_std": float(
                    frame[
                        "test_macro_f1"
                    ].std(
                        ddof=1
                    )
                ),
                "balanced_accuracy_mean": float(
                    frame[
                        "test_balanced_accuracy"
                    ].mean()
                ),
                "attack_recall_mean": float(
                    frame[
                        "test_attack_recall"
                    ].mean()
                ),
                "attack_f1_mean": float(
                    frame[
                        "test_attack_f1"
                    ].mean()
                ),
                "roc_auc_mean": float(
                    frame[
                        "test_roc_auc"
                    ].mean()
                ),
                "fpr_mean": float(
                    frame[
                        "test_fpr"
                    ].mean()
                ),
                "fnr_mean": float(
                    frame[
                        "test_fnr"
                    ].mean()
                )
            }
        )

    binary_summary = pd.DataFrame(
        binary_summary_rows
    )

    binary_summary.to_csv(
        BINARY_SUMMARY_PATH,
        index=False
    )

    monday = day_summary[
        day_summary[
            "test_day"
        ]
        == "Monday"
    ][
        [
            "model",
            "test_fpr_mean",
            "test_fpr_std"
        ]
    ]

    binary_days = day_summary[
        day_summary[
            "test_attack"
        ]
        > 0
    ][
        [
            "model",
            "test_day",
            "test_macro_f1_mean",
            "test_macro_f1_std",
            "test_attack_recall_mean",
            "test_attack_recall_std",
            "test_fpr_mean",
            "test_fnr_mean",
            "test_roc_auc_mean"
        ]
    ]

    print(
        "===== MONDAY BENIGN-ONLY CHALLENGE ====="
    )

    print(
        monday.to_string(
            index=False
        )
    )

    print()
    print(
        "===== BINARY DAY-HOLDOUT RESULTS ====="
    )

    print(
        binary_days.to_string(
            index=False
        )
    )

    print()
    print(
        "===== OVERALL BINARY DAY-HOLDOUT SUMMARY ====="
    )

    print(
        binary_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"RUNS={RUNS_PATH}"
    )

    print(
        f"DAY_SUMMARY={DAY_SUMMARY_PATH}"
    )

    print(
        f"BINARY_SUMMARY={BINARY_SUMMARY_PATH}"
    )

    print(
        "CICIDS_DAY_HOLDOUT_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()

