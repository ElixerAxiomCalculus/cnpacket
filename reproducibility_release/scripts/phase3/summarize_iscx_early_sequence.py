import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "iscx"
    / "early_sequence"
)

RUNS_PATH = (
    DIRECTORY
    / "early_sequence_runs.csv"
)

SUMMARY_PATH = (
    DIRECTORY
    / "early_sequence_summary.csv"
)

PAIRED_PATH = (
    DIRECTORY
    / "early_sequence_paired_deltas.csv"
)

DOMAIN_PATH = (
    DIRECTORY
    / "early_sequence_domain_summary.csv"
)

CLASS_PATH = (
    DIRECTORY
    / "early_sequence_class_summary.csv"
)


def main():
    reports = sorted(
        DIRECTORY.glob(
            "cnn_bilstm_n*_run_*_report.json"
        )
    )

    if len(reports) != 20:
        raise RuntimeError(
            f"Expected 20 reports, found {len(reports)}"
        )

    rows = []
    domain_rows = []
    class_rows = []

    cohort_hashes = set()

    for path in reports:
        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if int(
            report[
                "cohort_flows"
            ]
        ) != 1811:
            raise RuntimeError(
                f"Unexpected cohort size in {path}"
            )

        cohort_hashes.add(
            report[
                "cohort_sha256"
            ]
        )

        metrics = report[
            "test_metrics"
        ]

        rows.append(
            {
                "sequence_length": int(
                    report[
                        "sequence_length"
                    ]
                ),
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
            values = report[
                "test_domain_metrics"
            ][
                domain
            ]

            domain_rows.append(
                {
                    "sequence_length": int(
                        report[
                            "sequence_length"
                        ]
                    ),
                    "run": int(
                        report[
                            "run"
                        ]
                    ),
                    "domain": domain,
                    "samples": int(
                        values[
                            "samples"
                        ]
                    ),
                    "macro_f1": float(
                        values[
                            "macro_f1"
                        ]
                    ),
                    "balanced_accuracy": float(
                        values[
                            "balanced_accuracy"
                        ]
                    )
                }
            )

        for class_name, values in metrics[
            "per_class"
        ].items():
            class_rows.append(
                {
                    "sequence_length": int(
                        report[
                            "sequence_length"
                        ]
                    ),
                    "run": int(
                        report[
                            "run"
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
                    )
                }
            )

    if len(cohort_hashes) != 1:
        raise RuntimeError(
            f"Multiple cohort hashes found: {cohort_hashes}"
        )

    runs = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "sequence_length",
                "run"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_lengths = {
        5,
        10,
        20,
        50
    }

    if set(
        runs[
            "sequence_length"
        ]
    ) != expected_lengths:
        raise RuntimeError(
            "Sequence-length set mismatch"
        )

    counts = runs.groupby(
        "sequence_length"
    ).size()

    if not (
        counts == 5
    ).all():
        raise RuntimeError(
            f"Expected five folds per N: "
            f"{counts.to_dict()}"
        )

    for run, frame in runs.groupby(
        "run"
    ):
        if set(
            frame[
                "sequence_length"
            ]
        ) != expected_lengths:
            raise RuntimeError(
                f"Run {run} is not fully paired"
            )

        if frame[
            "test_fold"
        ].nunique() != 1:
            raise RuntimeError(
                f"Run {run} test-fold mismatch"
            )

    runs.to_csv(
        RUNS_PATH,
        index=False
    )

    summary_columns = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_roc_auc_ovr_macro",
        "test_average_precision_macro",
        "training_seconds",
        "test_us_per_sample",
        "peak_gpu_memory_mib",
        "best_epoch"
    ]

    summary_rows = []

    for sequence_length, frame in runs.groupby(
        "sequence_length",
        sort=True
    ):
        row = {
            "sequence_length": int(
                sequence_length
            ),
            "folds": len(
                frame
            )
        }

        for column in summary_columns:
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
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    pivot = runs.pivot(
        index="run",
        columns="sequence_length",
        values="test_macro_f1"
    )

    paired_rows = []

    baseline = pivot[
        5
    ]

    for sequence_length in [
        10,
        20,
        50
    ]:
        delta = (
            pivot[
                sequence_length
            ]
            - baseline
        )

        paired_rows.append(
            {
                "comparison": (
                    f"N{sequence_length}_minus_N5"
                ),
                "sequence_length": sequence_length,
                "mean_macro_f1_delta": float(
                    delta.mean()
                ),
                "std_macro_f1_delta": float(
                    delta.std(
                        ddof=1
                    )
                ),
                "min_macro_f1_delta": float(
                    delta.min()
                ),
                "max_macro_f1_delta": float(
                    delta.max()
                ),
                "improved_folds": int(
                    (
                        delta
                        > 0
                    ).sum()
                ),
                "worsened_folds": int(
                    (
                        delta
                        < 0
                    ).sum()
                ),
                "unchanged_folds": int(
                    (
                        delta
                        == 0
                    ).sum()
                )
            }
        )

    paired = pd.DataFrame(
        paired_rows
    )

    paired.to_csv(
        PAIRED_PATH,
        index=False
    )

    domains = pd.DataFrame(
        domain_rows
    )

    domain_summary = (
        domains.groupby(
            [
                "sequence_length",
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
                "sequence_length",
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
        "===== EARLY-CLASSIFICATION SUMMARY ====="
    )

    print(
        summary[
            [
                "sequence_length",
                "folds",
                "test_macro_f1_mean",
                "test_macro_f1_std",
                "test_balanced_accuracy_mean",
                "test_roc_auc_ovr_macro_mean",
                "test_accuracy_mean",
                "training_seconds_mean",
                "test_us_per_sample_mean",
                "peak_gpu_memory_mib_mean"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== PAIRED MACRO-F1 DELTAS VS N=5 ====="
    )

    print(
        paired.to_string(
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
        f"COHORT_SHA256="
        f"{next(iter(cohort_hashes))}"
    )

    print(
        f"RUNS={RUNS_PATH}"
    )

    print(
        f"SUMMARY={SUMMARY_PATH}"
    )

    print(
        f"PAIRED={PAIRED_PATH}"
    )

    print(
        f"DOMAIN_SUMMARY={DOMAIN_PATH}"
    )

    print(
        f"CLASS_SUMMARY={CLASS_PATH}"
    )

    print(
        "ISCX_EARLY_SEQUENCE_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()
