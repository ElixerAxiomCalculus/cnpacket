import json
import re
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "interim" / "cicids2017" / "raw_csv"
STAGE = ROOT / "data" / "interim" / "cicids2017" / "stage"
PROCESSED = ROOT / "data" / "processed" / "cicids2017"
ARTIFACTS = ROOT / "artifacts" / "phase2"
DB_PATH = ROOT / "data" / "interim" / "cicids2017" / "cicids2017_work.duckdb"

CHUNK_SIZE = 100_000
SPLIT_SEED = "20261011"

LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DOS_HULK",
    "PortScan": "PORTSCAN",
    "DDoS": "DDOS",
    "DoS GoldenEye": "DOS_GOLDENEYE",
    "FTP-Patator": "FTP_PATATOR",
    "SSH-Patator": "SSH_PATATOR",
    "DoS slowloris": "DOS_SLOWLORIS",
    "DoS Slowhttptest": "DOS_SLOWHTTPTEST",
    "Bot": "BOT",
    "Web Attack \ufffd Brute Force": "WEB_ATTACK_BRUTE_FORCE",
    "Web Attack \ufffd XSS": "WEB_ATTACK_XSS",
    "Infiltration": "INFILTRATION",
    "Web Attack \ufffd Sql Injection": "WEB_ATTACK_SQL_INJECTION",
    "Heartbleed": "HEARTBLEED"
}

DAY_ORDER = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4
}


def source_day(filename):
    for day in DAY_ORDER:
        if filename.lower().startswith(day.lower()):
            return day

    raise RuntimeError(
        f"Unable to derive source day from {filename}"
    )


def sanitize_name(value):
    value = str(value).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value
    ).strip("_")

    return value or "column"


def unique_names(values):
    counts = {}
    result = []

    for value in values:
        base = sanitize_name(value)
        counts[base] = counts.get(base, 0) + 1

        if counts[base] == 1:
            result.append(base)
        else:
            result.append(
                f"{base}_{counts[base]}"
            )

    return result


def quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def main():
    csv_files = sorted(
        path
        for path in RAW_CSV.rglob("*.csv")
        if path.is_file()
    )

    if len(csv_files) != 8:
        raise RuntimeError(
            f"Expected 8 CICIDS CSV files, found {len(csv_files)}"
        )

    STAGE.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in STAGE.glob("*.parquet"):
        path.unlink()

    first_header = pd.read_csv(
        csv_files[0],
        nrows=0
    )

    original_columns = [
        str(column).strip()
        for column in first_header.columns
    ]

    clean_columns = unique_names(
        original_columns
    )

    label_positions = [
        index
        for index, column in enumerate(original_columns)
        if column.lower() == "label"
    ]

    if label_positions != [len(original_columns) - 1]:
        raise RuntimeError(
            f"Unexpected Label position: {label_positions}"
        )

    label_column = clean_columns[
        label_positions[0]
    ]

    feature_columns = [
        column
        for column in clean_columns
        if column != label_column
    ]

    column_mapping = [
        {
            "position": index,
            "original": original,
            "clean": clean
        }
        for index, (original, clean) in enumerate(
            zip(
                original_columns,
                clean_columns
            )
        )
    ]

    (
        ARTIFACTS
        / "cicids_column_mapping.json"
    ).write_text(
        json.dumps(
            column_mapping,
            indent=2
        )
        + "\n"
    )

    (
        ARTIFACTS
        / "cicids_label_mapping.json"
    ).write_text(
        json.dumps(
            LABEL_MAP,
            indent=2,
            ensure_ascii=True
        )
        + "\n"
    )

    file_stats = []

    total_input_rows = 0
    total_invalid_rows = 0
    total_nan_cells = 0
    total_inf_cells = 0

    for path in csv_files:
        day = source_day(
            path.name
        )

        stage_path = (
            STAGE
            / f"{path.stem}.parquet"
        )

        writer = None
        source_offset = 0

        file_input_rows = 0
        file_invalid_rows = 0
        file_nan_cells = 0
        file_inf_cells = 0
        file_valid_rows = 0

        try:
            for chunk in pd.read_csv(
                path,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ):
                if len(chunk.columns) != len(clean_columns):
                    raise RuntimeError(
                        f"Schema length mismatch in {path}"
                    )

                chunk.columns = clean_columns

                raw_labels = (
                    chunk[label_column]
                    .astype("string")
                    .fillna("<MISSING>")
                    .str.strip()
                )

                canonical_labels = raw_labels.map(
                    LABEL_MAP
                )

                if canonical_labels.isna().any():
                    unknown = sorted(
                        set(
                            raw_labels[
                                canonical_labels.isna()
                            ].tolist()
                        )
                    )

                    raise RuntimeError(
                        f"Unknown labels in {path}: {unknown}"
                    )

                features = chunk[
                    feature_columns
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                ).astype(
                    "float64"
                )

                values = features.to_numpy(
                    dtype=np.float64,
                    copy=False
                )

                nan_mask = np.isnan(values)
                inf_mask = np.isinf(values)

                invalid_mask = (
                    nan_mask.any(axis=1)
                    | inf_mask.any(axis=1)
                )

                valid_mask = ~invalid_mask

                rows = len(chunk)

                invalid_rows = int(
                    invalid_mask.sum()
                )

                valid_rows = int(
                    valid_mask.sum()
                )

                nan_cells = int(
                    nan_mask.sum()
                )

                inf_cells = int(
                    inf_mask.sum()
                )

                source_rows = np.arange(
                    source_offset,
                    source_offset + rows,
                    dtype=np.int64
                )

                output = features.loc[
                    valid_mask
                ].reset_index(
                    drop=True
                )

                selected_labels = canonical_labels.loc[
                    valid_mask
                ].reset_index(
                    drop=True
                )

                output[
                    "canonical_label"
                ] = selected_labels.astype(
                    "string"
                )

                output[
                    "binary_label"
                ] = (
                    selected_labels
                    != "BENIGN"
                ).astype(
                    "int8"
                )

                output[
                    "source_file"
                ] = path.name

                output[
                    "source_day"
                ] = day

                output[
                    "source_day_order"
                ] = np.int8(
                    DAY_ORDER[day]
                )

                output[
                    "source_row"
                ] = source_rows[
                    valid_mask
                ]

                table = pa.Table.from_pandas(
                    output,
                    preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        stage_path,
                        table.schema,
                        compression="zstd"
                    )

                writer.write_table(
                    table
                )

                source_offset += rows

                file_input_rows += rows
                file_invalid_rows += invalid_rows
                file_valid_rows += valid_rows
                file_nan_cells += nan_cells
                file_inf_cells += inf_cells

        finally:
            if writer is not None:
                writer.close()

        file_stats.append(
            {
                "file": path.name,
                "day": day,
                "input_rows": file_input_rows,
                "invalid_rows_dropped": file_invalid_rows,
                "valid_rows_staged": file_valid_rows,
                "nan_cells": file_nan_cells,
                "inf_cells": file_inf_cells
            }
        )

        total_input_rows += file_input_rows
        total_invalid_rows += file_invalid_rows
        total_nan_cells += file_nan_cells
        total_inf_cells += file_inf_cells

        print(
            f"STAGED={path.name} "
            f"INPUT={file_input_rows} "
            f"INVALID={file_invalid_rows} "
            f"VALID={file_valid_rows}"
        )

    if DB_PATH.exists():
        DB_PATH.unlink()

    temp_directory = (
        ROOT
        / "data"
        / "interim"
        / "cicids2017"
        / "duckdb_temp"
    )

    if temp_directory.exists():
        shutil.rmtree(
            temp_directory
        )

    temp_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = duckdb.connect(
        str(DB_PATH)
    )

    connection.execute(
        "SET memory_limit='2GB'"
    )

    temp_sql = str(
        temp_directory
    ).replace(
        "'",
        "''"
    )

    connection.execute(
        f"SET temp_directory='{temp_sql}'"
    )

    stage_glob = str(
        STAGE / "*.parquet"
    ).replace(
        "'",
        "''"
    )

    connection.execute(
        f"""
        CREATE VIEW staged AS
        SELECT *
        FROM read_parquet('{stage_glob}')
        """
    )

    fingerprint_parts = ", ".join(
        f"CAST({quote_identifier(column)} AS VARCHAR)"
        for column in feature_columns
    )

    fingerprint_expression = (
        "md5(concat_ws('|', "
        + fingerprint_parts
        + "))"
    )

    connection.execute(
        f"""
        CREATE TABLE fingerprinted AS
        SELECT
            *,
            {fingerprint_expression}
                AS feature_fingerprint
        FROM staged
        """
    )

    staged_rows = connection.execute(
        "SELECT COUNT(*) FROM fingerprinted"
    ).fetchone()[0]

    connection.execute(
        """
        CREATE TABLE conflict_fingerprints AS
        SELECT feature_fingerprint
        FROM fingerprinted
        GROUP BY feature_fingerprint
        HAVING COUNT(DISTINCT canonical_label) > 1
        """
    )

    conflict_feature_groups = connection.execute(
        "SELECT COUNT(*) FROM conflict_fingerprints"
    ).fetchone()[0]

    conflict_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM fingerprinted
        WHERE feature_fingerprint IN (
            SELECT feature_fingerprint
            FROM conflict_fingerprints
        )
        """
    ).fetchone()[0]

    connection.execute(
        """
        CREATE TABLE nonconflict AS
        SELECT *
        FROM fingerprinted
        WHERE feature_fingerprint NOT IN (
            SELECT feature_fingerprint
            FROM conflict_fingerprints
        )
        """
    )

    nonconflict_rows = connection.execute(
        "SELECT COUNT(*) FROM nonconflict"
    ).fetchone()[0]

    metadata_columns = [
        "canonical_label",
        "binary_label",
        "source_file",
        "source_day",
        "source_day_order",
        "source_row",
        "feature_fingerprint"
    ]

    dedup_columns = (
        feature_columns
        + metadata_columns
    )

    dedup_select = ", ".join(
        quote_identifier(column)
        for column in dedup_columns
    )

    connection.execute(
        f"""
        CREATE TABLE deduplicated AS
        SELECT {dedup_select}
        FROM (
            SELECT
                {dedup_select},
                ROW_NUMBER() OVER (
                    PARTITION BY
                        feature_fingerprint,
                        canonical_label
                    ORDER BY
                        source_day_order,
                        source_file,
                        source_row
                ) AS duplicate_rank
            FROM nonconflict
        )
        WHERE duplicate_rank = 1
        """
    )

    deduplicated_rows = connection.execute(
        "SELECT COUNT(*) FROM deduplicated"
    ).fetchone()[0]

    duplicate_rows_removed = (
        nonconflict_rows
        - deduplicated_rows
    )

    ranked_columns = (
        feature_columns
        + metadata_columns
    )

    ranked_select = ", ".join(
        quote_identifier(column)
        for column in ranked_columns
    )

    connection.execute(
        f"""
        CREATE TABLE ranked AS
        SELECT
            {ranked_select},
            source_file
                || ':'
                || CAST(source_row AS VARCHAR)
                AS row_id,
            ROW_NUMBER() OVER (
                PARTITION BY binary_label
                ORDER BY md5(
                    source_file
                    || ':'
                    || CAST(source_row AS VARCHAR)
                    || '|{SPLIT_SEED}'
                )
            ) AS split_rank,
            COUNT(*) OVER (
                PARTITION BY binary_label
            ) AS class_count
        FROM deduplicated
        """
    )

    output_columns = (
        feature_columns
        + [
            "canonical_label",
            "binary_label",
            "source_file",
            "source_day",
            "source_day_order",
            "source_row",
            "feature_fingerprint",
            "row_id"
        ]
    )

    output_select = ", ".join(
        quote_identifier(column)
        for column in output_columns
    )

    connection.execute(
        f"""
        CREATE TABLE final_dataset AS
        SELECT
            {output_select},
            CASE
                WHEN split_rank
                    <= FLOOR(class_count * 0.70)
                    THEN 'train'
                WHEN split_rank
                    <= FLOOR(class_count * 0.85)
                    THEN 'validation'
                ELSE 'test'
            END AS benchmark_split
        FROM ranked
        """
    )

    final_rows = connection.execute(
        "SELECT COUNT(*) FROM final_dataset"
    ).fetchone()[0]

    final_path = (
        PROCESSED
        / "cicids2017_clean.parquet"
    )

    assignments_path = (
        ARTIFACTS
        / "cicids_split_assignments.parquet"
    )

    final_sql_path = str(
        final_path
    ).replace(
        "'",
        "''"
    )

    assignments_sql_path = str(
        assignments_path
    ).replace(
        "'",
        "''"
    )

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM final_dataset
        )
        TO '{final_sql_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    connection.execute(
        f"""
        COPY (
            SELECT
                row_id,
                binary_label,
                canonical_label,
                source_file,
                source_day,
                benchmark_split
            FROM final_dataset
        )
        TO '{assignments_sql_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    canonical_distribution = connection.execute(
        """
        SELECT
            canonical_label,
            COUNT(*) AS count
        FROM final_dataset
        GROUP BY canonical_label
        ORDER BY count DESC, canonical_label
        """
    ).df()

    binary_distribution = connection.execute(
        """
        SELECT
            binary_label,
            COUNT(*) AS count
        FROM final_dataset
        GROUP BY binary_label
        ORDER BY binary_label
        """
    ).df()

    split_distribution = connection.execute(
        """
        SELECT
            benchmark_split,
            binary_label,
            COUNT(*) AS count
        FROM final_dataset
        GROUP BY benchmark_split, binary_label
        ORDER BY benchmark_split, binary_label
        """
    ).df()

    day_distribution = connection.execute(
        """
        SELECT
            source_day,
            binary_label,
            COUNT(*) AS count
        FROM final_dataset
        GROUP BY
            source_day,
            binary_label
        ORDER BY
            MIN(source_day_order),
            binary_label
        """
    ).df()

    canonical_distribution.to_csv(
        ARTIFACTS
        / "cicids_clean_class_distribution.csv",
        index=False
    )

    binary_distribution.to_csv(
        ARTIFACTS
        / "cicids_binary_distribution.csv",
        index=False
    )

    split_distribution.to_csv(
        ARTIFACTS
        / "cicids_benchmark_split_summary.csv",
        index=False
    )

    day_distribution.to_csv(
        ARTIFACTS
        / "cicids_day_distribution.csv",
        index=False
    )

    pd.DataFrame(
        file_stats
    ).to_csv(
        ARTIFACTS
        / "cicids_cleaning_file_summary.csv",
        index=False
    )

    day_protocol = {
        "dataset": "CICIDS2017",
        "protocol_version": 1,
        "grouping_unit": "capture_day",
        "primary_target": "binary_label",
        "benign_label": 0,
        "attack_label": 1,
        "runs": [
            {
                "run": 0,
                "test_days": ["Monday"],
                "validation_days": ["Tuesday"],
                "training_days": [
                    "Wednesday",
                    "Thursday",
                    "Friday"
                ],
                "evaluation_scope": "benign_only_false_positive_challenge"
            },
            {
                "run": 1,
                "test_days": ["Tuesday"],
                "validation_days": ["Wednesday"],
                "training_days": [
                    "Monday",
                    "Thursday",
                    "Friday"
                ],
                "evaluation_scope": "binary_day_holdout"
            },
            {
                "run": 2,
                "test_days": ["Wednesday"],
                "validation_days": ["Thursday"],
                "training_days": [
                    "Monday",
                    "Tuesday",
                    "Friday"
                ],
                "evaluation_scope": "binary_day_holdout"
            },
            {
                "run": 3,
                "test_days": ["Thursday"],
                "validation_days": ["Friday"],
                "training_days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday"
                ],
                "evaluation_scope": "binary_day_holdout"
            },
            {
                "run": 4,
                "test_days": ["Friday"],
                "validation_days": ["Tuesday"],
                "training_days": [
                    "Monday",
                    "Wednesday",
                    "Thursday"
                ],
                "evaluation_scope": "binary_day_holdout"
            }
        ],
        "normalization_policy": (
            "Fit all scalers and feature transformations "
            "on training data only"
        ),
        "imbalance_policy": (
            "No preprocessing downsampling. "
            "Class weighting or sampling is applied "
            "only inside the training partition."
        )
    }

    (
        ARTIFACTS
        / "cicids_day_holdout_protocol.json"
    ).write_text(
        json.dumps(
            day_protocol,
            indent=2
        )
        + "\n"
    )

    report = {
        "dataset": "CICIDS2017",
        "protocol_version": 1,
        "split_seed": SPLIT_SEED,
        "feature_count": len(feature_columns),
        "input_rows": total_input_rows,
        "nan_cells_detected": total_nan_cells,
        "infinite_cells_detected": total_inf_cells,
        "invalid_rows_dropped": total_invalid_rows,
        "valid_rows_staged": staged_rows,
        "conflicting_feature_groups": conflict_feature_groups,
        "conflicting_rows_dropped": conflict_rows,
        "duplicate_rows_dropped": duplicate_rows_removed,
        "final_rows": final_rows,
        "binary_definition": {
            "0": "BENIGN",
            "1": "ATTACK"
        },
        "benchmark_split": {
            "method": (
                "Deterministic class-stratified ordering "
                "by MD5 of stable source row identifier"
            ),
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15
        },
        "day_holdout": {
            "grouping_unit": "capture_day",
            "runs": 5
        },
        "label_repair": (
            "The raw files contain the UTF-8 replacement "
            "character U+FFFD in the three Web Attack labels. "
            "Those exact byte-level labels are mapped "
            "deterministically to ASCII canonical names."
        )
    }

    (
        ARTIFACTS
        / "cicids_cleaning_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
        + "\n"
    )

    connection.close()

    print()
    print("CICIDS_PREPARATION_COMPLETE")
    print(
        f"INPUT_ROWS={total_input_rows}"
    )
    print(
        f"NAN_CELLS={total_nan_cells}"
    )
    print(
        f"INF_CELLS={total_inf_cells}"
    )
    print(
        f"INVALID_ROWS_DROPPED={total_invalid_rows}"
    )
    print(
        f"CONFLICTING_FEATURE_GROUPS={conflict_feature_groups}"
    )
    print(
        f"CONFLICTING_ROWS_DROPPED={conflict_rows}"
    )
    print(
        f"DUPLICATE_ROWS_DROPPED={duplicate_rows_removed}"
    )
    print(
        f"FINAL_ROWS={final_rows}"
    )
    print(
        f"FEATURES={len(feature_columns)}"
    )
    print(
        "BINARY_TARGET=BENIGN_VS_ATTACK"
    )
    print(
        "BENCHMARK_SPLIT=70_15_15_STRATIFIED"
    )
    print(
        "DAY_HOLDOUT_PROTOCOL=PASS"
    )
    print(
        f"OUTPUT={final_path}"
    )


if __name__ == "__main__":
    main()