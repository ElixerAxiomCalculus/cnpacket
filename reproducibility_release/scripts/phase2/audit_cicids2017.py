import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "cicids2017" / "raw_csv"
OUTPUT = ROOT / "artifacts" / "phase2"

CHUNK_SIZE = 100_000


def normalize_columns(columns):
    return [
        str(column).strip()
        for column in columns
    ]


def find_csv_files():
    return sorted(
        path
        for path in INPUT.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".csv"
    )


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    files = find_csv_files()

    if not files:
        raise RuntimeError(
            f"No CSV files found under {INPUT}"
        )

    global_labels = Counter()
    file_rows = []
    canonical_columns = None
    total_rows = 0

    for path in files:
        header = pd.read_csv(
            path,
            nrows=0
        )

        columns = normalize_columns(
            header.columns
        )

        if "Label" not in columns:
            raise RuntimeError(
                f"Label column not found in {path}"
            )

        if canonical_columns is None:
            canonical_columns = columns

        same_schema = (
            columns == canonical_columns
        )

        row_count = 0
        labels = Counter()

        for chunk in pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            chunk.columns = normalize_columns(
                chunk.columns
            )

            label_values = (
                chunk["Label"]
                .astype("string")
                .fillna("<MISSING>")
                .str.strip()
            )

            counts = Counter(
                label_values.tolist()
            )

            labels.update(counts)
            global_labels.update(counts)

            rows = len(chunk)
            row_count += rows
            total_rows += rows

        file_rows.append(
            {
                "file": str(
                    path.relative_to(ROOT)
                ),
                "rows": row_count,
                "columns": len(columns),
                "same_schema_as_first_file": same_schema,
                "distinct_labels": len(labels),
                "labels": dict(
                    sorted(labels.items())
                )
            }
        )

        print(
            f"FILE={path.name} "
            f"ROWS={row_count} "
            f"COLUMNS={len(columns)} "
            f"LABELS={len(labels)} "
            f"SCHEMA_MATCH={str(same_schema).upper()}"
        )

    class_rows = []

    for label, count in sorted(
        global_labels.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    ):
        class_rows.append(
            {
                "label": label,
                "count": count,
                "percentage": round(
                    100.0
                    * count
                    / total_rows,
                    6
                )
            }
        )

    class_csv = (
        OUTPUT
        / "cicids_class_distribution.csv"
    )

    with class_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "count",
                "percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(
            class_rows
        )

    file_csv = (
        OUTPUT
        / "cicids_file_summary.csv"
    )

    with file_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "rows",
                "columns",
                "same_schema_as_first_file",
                "distinct_labels"
            ]
        )

        writer.writeheader()

        for row in file_rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in writer.fieldnames
                }
            )

    audit = {
        "dataset": "CICIDS2017",
        "csv_files": len(files),
        "total_rows": total_rows,
        "feature_columns_including_label": (
            len(canonical_columns)
            if canonical_columns
            else 0
        ),
        "distinct_labels": len(
            global_labels
        ),
        "schema_consistent": all(
            row[
                "same_schema_as_first_file"
            ]
            for row in file_rows
        ),
        "columns": canonical_columns,
        "class_distribution": class_rows,
        "files": file_rows
    }

    audit_json = (
        OUTPUT
        / "cicids_audit.json"
    )

    audit_json.write_text(
        json.dumps(
            audit,
            indent=2
        )
        + "\n"
    )

    print()
    print(
        f"CSV_FILES={len(files)}"
    )

    print(
        f"TOTAL_ROWS={total_rows}"
    )

    print(
        f"COLUMNS={len(canonical_columns)}"
    )

    print(
        f"DISTINCT_LABELS={len(global_labels)}"
    )

    print(
        "SCHEMA_CONSISTENT="
        f"{str(audit['schema_consistent']).upper()}"
    )

    print()
    print("CLASS_DISTRIBUTION")

    for row in class_rows:
        print(
            f"{row['label']}: "
            f"{row['count']} "
            f"({row['percentage']:.6f}%)"
        )

    print()
    print(
        f"AUDIT_JSON={audit_json}"
    )

    print(
        f"CLASS_CSV={class_csv}"
    )

    print(
        f"FILE_CSV={file_csv}"
    )


if __name__ == "__main__":
    main()