import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set"
)

PHASE2_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "ustc_protocol.json"
)

ZERO_RUNS_PATH = (
    DIRECTORY
    / "zero_shot_runs.csv"
)

ZERO_SUMMARY_PATH = (
    DIRECTORY
    / "zero_shot_summary.csv"
)

PAIRED_PATH = (
    DIRECTORY
    / "zero_shot_paired_by_target.csv"
)

PAIRED_SUMMARY_PATH = (
    DIRECTORY
    / "zero_shot_paired_summary.csv"
)

FEW_RUNS_PATH = (
    DIRECTORY
    / "few_shot_runs.csv"
)

FEW_SUMMARY_PATH = (
    DIRECTORY
    / "few_shot_summary.csv"
)


def aggregate(
    frame,
    group_columns,
    metric_columns
):
    rows = []

    for key, group in frame.groupby(
        group_columns,
        sort=True
    ):
        if not isinstance(
            key,
            tuple
        ):
            key = (
                key,
            )

        row = {
            column: value
            for column, value in zip(
                group_columns,
                key
            )
        }

        row[
            "runs"
        ] = len(
            group
        )

        for column in metric_columns:
            row[
                f"{column}_mean"
            ] = float(
                group[
                    column
                ].mean()
            )

            row[
                f"{column}_std"
            ] = float(
                group[
                    column
                ].std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main():
    phase2 = json.loads(
        PHASE2_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    rotations = {
        int(
            item[
                "run"
            ]
        ): item
        for item in phase2[
            "open_set"
        ][
            "runs_detail"
        ]
    }

    reports = sorted(
        DIRECTORY.glob(
            "*_run_*_report.json"
        )
    )

    expected = {
        (
            method,
            run
        )
        for method in [
            "ce_msp",
            "supcon_proto"
        ]
        for run in range(
            10
        )
    }

    observed = set()

    zero_rows = []
    few_rows = []

    for path in reports:
        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if report.get(
            "experiment"
        ) != "ustc_tfc2016_open_set":
            continue

        method = report[
            "method"
        ]

        run = int(
            report[
                "run"
            ]
        )

        if method not in {
            "ce_msp",
            "supcon_proto"
        }:
            continue

        key = (
            method,
            run
        )

        if key in observed:
            raise RuntimeError(
                f"Duplicate report {key}"
            )

        observed.add(
            key
        )

        rotation = rotations[
            run
        ]

        if report[
            "target_unknown_family"
        ] != rotation[
            "target_unknown_family"
        ]:
            raise RuntimeError(
                f"Target mismatch {key}"
            )

        if report[
            "calibration_unknown_family"
        ] != rotation[
            "calibration_unknown_family"
        ]:
            raise RuntimeError(
                f"Calibration mismatch {key}"
            )

        zero = report[
            "zero_shot"
        ]

        calibration = report[
            "threshold_calibration"
        ]

        zero_rows.append(
            {
                "method": method,
                "run": run,
                "target_family": report[
                    "target_unknown_family"
                ],
                "calibration_family": report[
                    "calibration_unknown_family"
                ],
                "known_train": report[
                    "partition_counts"
                ][
                    "known_train"
                ],
                "known_validation": report[
                    "partition_counts"
                ][
                    "known_validation"
                ],
                "known_test": report[
                    "partition_counts"
                ][
                    "known_test"
                ],
                "target_query": report[
                    "partition_counts"
                ][
                    "target_query_pool"
                ],
                "best_known_validation_macro_f1": report[
                    "best_known_validation_macro_f1"
                ],
                "calibration_balanced_accuracy": calibration[
                    "balanced_accuracy"
                ],
                "calibration_unknown_recall": calibration[
                    "calibration_unknown_recall"
                ],
                "calibration_known_frr": calibration[
                    "known_false_rejection_rate"
                ],
                "threshold": calibration[
                    "threshold"
                ],
                "unknown_auroc": zero[
                    "unknown_detection_auroc"
                ],
                "unknown_aupr": zero[
                    "unknown_detection_average_precision"
                ],
                "known_frr": zero[
                    "known_false_rejection_rate"
                ],
                "unknown_recall": zero[
                    "unknown_recall"
                ],
                "known_macro_f1_without_rejection": zero[
                    "known_family_macro_f1_without_rejection"
                ],
                "open_set_macro_f1": zero[
                    "open_set"
                ][
                    "macro_f1"
                ],
                "open_set_balanced_accuracy": zero[
                    "open_set"
                ][
                    "balanced_accuracy"
                ],
                "open_set_accuracy": zero[
                    "open_set"
                ][
                    "accuracy"
                ],
                "training_seconds": report[
                    "training_seconds"
                ],
                "peak_gpu_memory_mib": report[
                    "peak_gpu_memory_mib"
                ],
                "model_size_bytes": report[
                    "model_size_bytes"
                ]
            }
        )

        if method == "supcon_proto":
            few = report[
                "few_shot"
            ]

            for k in [
                1,
                5,
                10
            ]:
                values = few[
                    str(
                        k
                    )
                ]

                classification = values[
                    "nine_family_without_rejection"
                ]

                target_values = classification[
                    "per_class"
                ][
                    report[
                        "target_unknown_family"
                    ]
                ]

                threshold = values[
                    "threshold"
                ]

                few_rows.append(
                    {
                        "run": run,
                        "target_family": report[
                            "target_unknown_family"
                        ],
                        "k": k,
                        "nine_family_macro_f1": classification[
                            "macro_f1"
                        ],
                        "nine_family_balanced_accuracy": classification[
                            "balanced_accuracy"
                        ],
                        "nine_family_accuracy": classification[
                            "accuracy"
                        ],
                        "target_precision": target_values[
                            "precision"
                        ],
                        "target_recall": target_values[
                            "recall"
                        ],
                        "target_f1": target_values[
                            "f1"
                        ],
                        "target_correct_classification_rate": values[
                            "target_query_correct_classification_rate"
                        ],
                        "target_acceptance_rate": values[
                            "target_query_acceptance_rate"
                        ],
                        "target_correct_and_accepted_rate": values[
                            "target_query_correct_and_accepted_rate"
                        ],
                        "thresholded_nine_family_macro_f1": values[
                            "thresholded_nine_family_macro_f1"
                        ],
                        "adapted_threshold": threshold[
                            "threshold"
                        ],
                        "adapted_calibration_balanced_accuracy": threshold[
                            "balanced_accuracy"
                        ],
                        "adapted_calibration_unknown_recall": threshold[
                            "calibration_unknown_recall"
                        ],
                        "adapted_calibration_known_frr": threshold[
                            "known_false_rejection_rate"
                        ]
                    }
                )

    if observed != expected:
        missing = sorted(
            expected
            - observed
        )

        extra = sorted(
            observed
            - expected
        )

        raise RuntimeError(
            f"Report set mismatch "
            f"missing={missing} extra={extra}"
        )

    zero_runs = (
        pd.DataFrame(
            zero_rows
        )
        .sort_values(
            [
                "method",
                "run"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    zero_runs.to_csv(
        ZERO_RUNS_PATH,
        index=False
    )

    zero_metrics = [
        "best_known_validation_macro_f1",
        "calibration_balanced_accuracy",
        "calibration_unknown_recall",
        "calibration_known_frr",
        "unknown_auroc",
        "unknown_aupr",
        "known_frr",
        "unknown_recall",
        "known_macro_f1_without_rejection",
        "open_set_macro_f1",
        "open_set_balanced_accuracy",
        "open_set_accuracy",
        "training_seconds",
        "peak_gpu_memory_mib",
        "model_size_bytes"
    ]

    zero_summary = aggregate(
        zero_runs,
        [
            "method"
        ],
        zero_metrics
    )

    zero_summary.to_csv(
        ZERO_SUMMARY_PATH,
        index=False
    )

    ce = (
        zero_runs[
            zero_runs[
                "method"
            ]
            == "ce_msp"
        ]
        .set_index(
            "run"
        )
    )

    supcon = (
        zero_runs[
            zero_runs[
                "method"
            ]
            == "supcon_proto"
        ]
        .set_index(
            "run"
        )
    )

    paired_rows = []

    paired_metrics = [
        "best_known_validation_macro_f1",
        "calibration_balanced_accuracy",
        "unknown_auroc",
        "unknown_aupr",
        "known_frr",
        "unknown_recall",
        "known_macro_f1_without_rejection",
        "open_set_macro_f1",
        "open_set_balanced_accuracy"
    ]

    for run in range(
        10
    ):
        row = {
            "run": run,
            "target_family": ce.loc[
                run,
                "target_family"
            ],
            "calibration_family": ce.loc[
                run,
                "calibration_family"
            ]
        }

        for metric in paired_metrics:
            row[
                f"ce_{metric}"
            ] = ce.loc[
                run,
                metric
            ]

            row[
                f"supcon_{metric}"
            ] = supcon.loc[
                run,
                metric
            ]

            row[
                f"delta_{metric}"
            ] = (
                supcon.loc[
                    run,
                    metric
                ]
                - ce.loc[
                    run,
                    metric
                ]
            )

        paired_rows.append(
            row
        )

    paired = pd.DataFrame(
        paired_rows
    )

    paired.to_csv(
        PAIRED_PATH,
        index=False
    )

    paired_summary_rows = []

    lower_is_better = {
        "known_frr"
    }

    for metric in paired_metrics:
        delta = paired[
            f"delta_{metric}"
        ]

        if metric in lower_is_better:
            supcon_wins = int(
                (
                    delta
                    < 0
                ).sum()
            )

            ce_wins = int(
                (
                    delta
                    > 0
                ).sum()
            )

        else:
            supcon_wins = int(
                (
                    delta
                    > 0
                ).sum()
            )

            ce_wins = int(
                (
                    delta
                    < 0
                ).sum()
            )

        paired_summary_rows.append(
            {
                "metric": metric,
                "mean_supcon_minus_ce": float(
                    delta.mean()
                ),
                "std_supcon_minus_ce": float(
                    delta.std(
                        ddof=1
                    )
                ),
                "supcon_wins": supcon_wins,
                "ce_wins": ce_wins,
                "ties": int(
                    (
                        delta
                        == 0
                    ).sum()
                )
            }
        )

    paired_summary = pd.DataFrame(
        paired_summary_rows
    )

    paired_summary.to_csv(
        PAIRED_SUMMARY_PATH,
        index=False
    )

    few_runs = (
        pd.DataFrame(
            few_rows
        )
        .sort_values(
            [
                "k",
                "run"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        few_runs
    ) != 30:
        raise RuntimeError(
            f"Expected 30 few-shot rows, "
            f"found {len(few_runs)}"
        )

    few_runs.to_csv(
        FEW_RUNS_PATH,
        index=False
    )

    few_metrics = [
        "nine_family_macro_f1",
        "nine_family_balanced_accuracy",
        "nine_family_accuracy",
        "target_precision",
        "target_recall",
        "target_f1",
        "target_correct_classification_rate",
        "target_acceptance_rate",
        "target_correct_and_accepted_rate",
        "thresholded_nine_family_macro_f1",
        "adapted_calibration_balanced_accuracy",
        "adapted_calibration_unknown_recall",
        "adapted_calibration_known_frr"
    ]

    few_summary = aggregate(
        few_runs,
        [
            "k"
        ],
        few_metrics
    )

    few_summary.to_csv(
        FEW_SUMMARY_PATH,
        index=False
    )

    print(
        "===== ZERO-SHOT PER ROTATION ====="
    )

    print(
        zero_runs[
            [
                "method",
                "run",
                "target_family",
                "calibration_family",
                "best_known_validation_macro_f1",
                "unknown_auroc",
                "unknown_aupr",
                "known_frr",
                "unknown_recall",
                "open_set_macro_f1"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== ZERO-SHOT SUMMARY ====="
    )

    print(
        zero_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PAIRED SUPCON - CE ====="
    )

    print(
        paired_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== FEW-SHOT PER TARGET ====="
    )

    print(
        few_runs[
            [
                "run",
                "target_family",
                "k",
                "nine_family_macro_f1",
                "target_f1",
                "target_correct_classification_rate",
                "target_correct_and_accepted_rate",
                "thresholded_nine_family_macro_f1"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== FEW-SHOT SUMMARY ====="
    )

    print(
        few_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"ZERO_RUNS={ZERO_RUNS_PATH}"
    )

    print(
        f"ZERO_SUMMARY={ZERO_SUMMARY_PATH}"
    )

    print(
        f"PAIRED={PAIRED_PATH}"
    )

    print(
        f"PAIRED_SUMMARY={PAIRED_SUMMARY_PATH}"
    )

    print(
        f"FEW_RUNS={FEW_RUNS_PATH}"
    )

    print(
        f"FEW_SUMMARY={FEW_SUMMARY_PATH}"
    )

    print(
        "USTC_OPEN_SET_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()
