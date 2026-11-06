from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
)

OUTPUT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "paper"
    / "experimental_completion"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

CANONICAL_FAMILIES = [
    "Cridex",
    "Geodo",
    "Htbot",
    "Miuref",
    "Neris",
    "Nsis-ay",
    "Shifu",
    "Tinba",
    "Virut",
    "Zeus"
]


def parse_family_order(
    value
):
    if (
        isinstance(
            value,
            list
        )
        and len(
            value
        )
        == 10
        and all(
            isinstance(
                item,
                str
            )
            for item in value
        )
    ):
        return list(
            value
        )

    if isinstance(
        value,
        dict
    ):
        keys = list(
            value.keys()
        )

        if (
            len(
                keys
            )
            == 10
            and set(
                keys
            )
            == set(
                CANONICAL_FAMILIES
            )
        ):
            return keys

    raise RuntimeError(
        f"Unsupported families field: {value!r}"
    )


def reorder_matrix(
    matrix,
    source_order
):
    if set(
        source_order
    ) != set(
        CANONICAL_FAMILIES
    ):
        raise RuntimeError(
            "Family set mismatch"
        )

    positions = {
        family: index
        for index, family in enumerate(
            source_order
        )
    }

    indices = [
        positions[
            family
        ]
        for family in CANONICAL_FAMILIES
    ]

    return matrix[
        np.ix_(
            indices,
            indices
        )
    ]


def normalize_rows(
    matrix
):
    row_totals = matrix.sum(
        axis=1,
        keepdims=True
    )

    if (
        row_totals
        <= 0
    ).any():
        raise RuntimeError(
            "Empty true-class row"
        )

    return (
        matrix
        / row_totals
    )


def main():
    reports = sorted(
        REPORT_DIRECTORY.glob(
            "cnn_bilstm_seed_*_report.json"
        )
    )

    if len(
        reports
    ) != 5:
        raise RuntimeError(
            f"Expected 5 reports, found {len(reports)}"
        )

    normalized_matrices = []

    source_rows = []

    for report_path in reports:
        report = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        if "test_metrics" not in report:
            raise RuntimeError(
                f"Missing test_metrics in {report_path.name}"
            )

        if (
            "confusion_matrix"
            not in report[
                "test_metrics"
            ]
        ):
            raise RuntimeError(
                f"Missing test confusion matrix in "
                f"{report_path.name}"
            )

        source_order = parse_family_order(
            report[
                "families"
            ]
        )

        matrix = np.asarray(
            report[
                "test_metrics"
            ][
                "confusion_matrix"
            ],
            dtype=np.int64
        )

        if matrix.shape != (
            10,
            10
        ):
            raise RuntimeError(
                f"Unexpected confusion shape "
                f"{matrix.shape} in {report_path.name}"
            )

        matrix = reorder_matrix(
            matrix,
            source_order
        )

        normalized = normalize_rows(
            matrix.astype(
                np.float64
            )
        )

        normalized_matrices.append(
            normalized
        )

        source_rows.append(
            {
                "report": report_path.name,
                "seed": report[
                    "seed"
                ],
                "test_rows": report[
                    "test_rows"
                ],
                "family_order": "|".join(
                    source_order
                )
            }
        )

    stack = np.stack(
        normalized_matrices,
        axis=0
    )

    mean_matrix = stack.mean(
        axis=0
    )

    std_matrix = stack.std(
        axis=0,
        ddof=1
    )

    mean_frame = pd.DataFrame(
        mean_matrix,
        index=CANONICAL_FAMILIES,
        columns=CANONICAL_FAMILIES
    )

    std_frame = pd.DataFrame(
        std_matrix,
        index=CANONICAL_FAMILIES,
        columns=CANONICAL_FAMILIES
    )

    mean_path = (
        OUTPUT_DIRECTORY
        / "closed_set_confusion_mean.csv"
    )

    std_path = (
        OUTPUT_DIRECTORY
        / "closed_set_confusion_std.csv"
    )

    source_path = (
        OUTPUT_DIRECTORY
        / "closed_set_confusion_sources.csv"
    )

    figure_path = (
        OUTPUT_DIRECTORY
        / "closed_set_confusion_mean.png"
    )

    mean_frame.to_csv(
        mean_path
    )

    std_frame.to_csv(
        std_path
    )

    pd.DataFrame(
        source_rows
    ).to_csv(
        source_path,
        index=False
    )

    fig, ax = plt.subplots(
        figsize=(
            8.4,
            7.2
        )
    )

    image = ax.imshow(
        mean_matrix,
        vmin=0.0,
        vmax=1.0
    )

    positions = np.arange(
        len(
            CANONICAL_FAMILIES
        )
    )

    ax.set_xticks(
        positions
    )

    ax.set_yticks(
        positions
    )

    ax.set_xticklabels(
        CANONICAL_FAMILIES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        CANONICAL_FAMILIES
    )

    ax.set_xlabel(
        "Predicted malware family"
    )

    ax.set_ylabel(
        "True malware family"
    )

    ax.set_title(
        "USTC-TFC2016 closed-set confusion matrix"
    )

    for row in range(
        10
    ):
        for column in range(
            10
        ):
            value = mean_matrix[
                row,
                column
            ]

            if (
                row == column
                or value >= 0.05
            ):
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center"
                )

    fig.colorbar(
        image,
        ax=ax,
        label="Mean row-normalized fraction"
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    diagonal = np.diag(
        mean_matrix
    )

    off_diagonal = mean_matrix.copy()

    np.fill_diagonal(
        off_diagonal,
        -1.0
    )

    print(
        "===== CLOSED-SET TEST CONFUSION ====="
    )

    print(
        f"REPORT_COUNT={len(reports)}"
    )

    print(
        f"SEED_COUNT={len(normalized_matrices)}"
    )

    print(
        f"MEAN_DIAGONAL_RECALL="
        f"{diagonal.mean():.8f}"
    )

    print()

    for family, recall in zip(
        CANONICAL_FAMILIES,
        diagonal
    ):
        print(
            f"FAMILY={family} "
            f"MEAN_RECALL="
            f"{recall:.8f}"
        )

    print()

    print(
        "===== DOMINANT MISCLASSIFICATION PER FAMILY ====="
    )

    for row, family in enumerate(
        CANONICAL_FAMILIES
    ):
        column = int(
            np.argmax(
                off_diagonal[
                    row
                ]
            )
        )

        value = float(
            off_diagonal[
                row,
                column
            ]
        )

        print(
            f"TRUE={family} "
            f"MOST_CONFUSED_WITH="
            f"{CANONICAL_FAMILIES[column]} "
            f"FRACTION={value:.8f}"
        )

    print()

    print(
        f"MEAN_CSV={mean_path}"
    )

    print(
        f"STD_CSV={std_path}"
    )

    print(
        f"SOURCES_CSV={source_path}"
    )

    print(
        f"FIGURE={figure_path}"
    )

    print(
        "CLOSED_SET_CONFUSION_MATRIX=PASS"
    )


if __name__ == "__main__":
    main()
