import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "data" / "interim" / "cicids2017" / "stage"
HASHED = ROOT / "data" / "interim" / "cicids2017" / "stage_hashed"
PROCESSED = ROOT / "data" / "processed" / "cicids2017"
ARTIFACTS = ROOT / "artifacts" / "phase2"
TEMP = ROOT / "data" / "interim" / "cicids2017" / "duckdb_temp_lowmem"
DB_PATH = ROOT / "data" / "interim" / "cicids2017" / "cicids2017_lowmem.duckdb"

BATCH_SIZE = 50_000
SPLIT_SEED = "20261011"

METADATA_COLUMNS = [
    "canonical_label",
    "binary_label",
    "source_file",
    "source_day",
    "source_day_order",
    "source_row"
]


def quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def prepare_hashed_stage():
    stage_files = sorted(
        STAGE.glob("*.parquet")
    )

    if len(stage_files) != 8:
        raise RuntimeError(
            f"Expected 8 staged Parquet files, found {len(stage_files)}"
        )

    if HASHED.exists():
        shutil.rmtree(HASHED)

    HASHED.mkdir(
        parents=True,
        exist_ok=True
    )

    first_file = pq.ParquetFile(
        stage_files[0]
    )

    columns = first_file.schema_arrow.names

    feature_columns = [
        column
        for column in columns
        if column not in METADATA_COLUMNS
    ]

    if len(feature_columns) != 78:
        raise RuntimeError(
            f"Expected 78 feature columns, found {len(feature_columns)}"
        )

    total_rows = 0

    for source_path in stage_files:
        destination = (
            HASHED
            / source_path.name
        )

        parquet_file = pq.ParquetFile(
            source_path
        )

        writer = None
        file_rows = 0

        try:
            for batch in parquet_file.iter_batches(
                batch_size=BATCH_SIZE
            ):
                frame = batch.to_pandas()

                hash_a = pd.util.hash_pandas_object(
                    frame[feature_columns],
                    index=False
                ).astype("uint64")

                hash_b = pd.util.hash_pandas_object(
                    frame[
                        list(
                            reversed(
                                feature_columns
                            )
                        )
                    ],
                    index=False
                ).astype("uint64")

                frame[
                    "feature_hash_a"
                ] = hash_a.to_numpy()

                frame[
                    "feature_hash_b"
                ] = hash_b.to_numpy()

                table = pa.Table.from_pandas(
                    frame,
                    preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        destination,
                        table.schema,
                        compression="zstd"
                    )

                writer.write_table(
                    table
                )

                file_rows += len(frame)
                total_rows += len(frame)

        finally:
            if writer is not None:
                writer.close()

        print(
            f"HASHED={source_path.name} "
            f"ROWS={file_rows}"
        )

    return feature_columns, total_rows


def read_input_row_count():
    audit_path = (
        ARTIFACTS
        / "cicids_audit.json"
    )

    if not audit_path.exists():
        return None

    audit = json.loads(
        audit_path.read_text()
    )

    return int(
        audit["total_rows"]
    )


def write_day_protocol():
    protocol = {
        "dataset": "CICIDS2017",
        "protocol_version": 1,
        "grouping_unit": "capture_day",
        "primary_target": "binary_label",
        "benign_label": 0,
        "attack_label": 1,
        "runs": [
            {
                "run": 0,
                "test_days": [
                    "Monday"
                ],
                "validation_days": [
                    "Tuesday"
                ],
                "training_days": [
                    "Wednesday",
                    "Thursday",
                    "Friday"
                ],
                "evaluation_scope": (
                    "benign_only_false_positive_challenge"
                )
            },
            {
                "run": 1,
                "test_days": [
                    "Tuesday"
                ],
                "validation_days": [
                    "Wednesday"
                ],
                "training_days": [
                    "Monday",
                    "Thursday",
                    "Friday"
                ],
                "evaluation_scope": (
                    "binary_day_holdout"
                )
            },
            {
                "run": 2,
                "test_days": [
                    "Wednesday"
                ],
                "validation_days": [
                    "Thursday"
                ],
                "training_days": [
                    "Monday",
                    "Tuesday",
                    "Friday"
                ],
                "evaluation_scope": (
                    "binary_day_holdout"
                )
            },
            {
                "run": 3,
                "test_days": [
                    "Thursday"
                ],
                "validation_days": [
                    "Friday"
                ],
                "training_days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday"
                ],
                "evaluation_scope": (
                    "binary_day_holdout"
                )
            },
            {
                "run": 4,
                "test_days": [
                    "Friday"
                ],
                "validation_days": [
                    "Tuesday"
                ],
                "training_days": [
                    "Monday",
                    "Wednesday",
                    "Thursday"
                ],
                "evaluation_scope": (
                    "binary_day_holdout"
                )
            }
        ],
        "normalization_policy": (
            "Fit every scaler and feature transformation "
            "using training data only"
        ),
        "imbalance_policy": (
            "No preprocessing downsampling. "
            "Class weighting or sampling is allowed "
            "only inside a training partition."
        )
    }

    (
        ARTIFACTS
        / "cicids_day_holdout_protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2
        )
        + "\n"
    )


def main():
    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    print("HASH_STAGE_BEGIN")

    (
        feature_columns,
        hashed_rows
    ) = prepare_hashed_stage()

    print(
        f"HASH_STAGE_ROWS={hashed_rows}"
    )

    if TEMP.exists():
        shutil.rmtree(TEMP)

    TEMP.mkdir(
        parents=True,
        exist_ok=True
    )

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = duckdb.connect(
        str(DB_PATH)
    )

    connection.execute(
        "SET memory_limit='1GB'"
    )

    connection.execute(
        "SET threads=1"
    )

    connection.execute(
        "SET preserve_insertion_order=false"
    )

    temp_path = str(
        TEMP
    ).replace(
        "'",
        "''"
    )

    connection.execute(
        f"SET temp_directory='{temp_path}'"
    )

    hashed_glob = str(
        HASHED / "*.parquet"
    ).replace(
        "'",
        "''"
    )

    connection.execute(
        f"""
        CREATE VIEW hashed AS
        SELECT *
        FROM read_parquet('{hashed_glob}')
        """
    )

    staged_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM hashed
        """
    ).fetchone()[0]

    if staged_rows != hashed_rows:
        raise RuntimeError(
            f"Hashed row mismatch: "
            f"{staged_rows} != {hashed_rows}"
        )

    print(
        f"STAGED_ROWS={staged_rows}"
    )

    print(
        "CONFLICT_SCAN_BEGIN"
    )

    connection.execute(
        """
        CREATE TABLE conflict_hashes AS
        SELECT
            feature_hash_a,
            feature_hash_b
        FROM hashed
        GROUP BY
            feature_hash_a,
            feature_hash_b
        HAVING
            COUNT(
                DISTINCT canonical_label
            ) > 1
        """
    )

    conflict_feature_groups = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM conflict_hashes
            """
        ).fetchone()[0]
    )

    conflict_rows = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM hashed h
            INNER JOIN conflict_hashes c
                ON
                    h.feature_hash_a
                        = c.feature_hash_a
                    AND
                    h.feature_hash_b
                        = c.feature_hash_b
            """
        ).fetchone()[0]
    )

    nonconflict_rows = (
        staged_rows
        - conflict_rows
    )

    print(
        f"CONFLICTING_FEATURE_GROUPS="
        f"{conflict_feature_groups}"
    )

    print(
        f"CONFLICTING_ROWS="
        f"{conflict_rows}"
    )

    print(
        "DEDUPLICATION_BEGIN"
    )

    connection.execute(
        """
        CREATE TABLE selected_rows AS
        SELECT
            source_file,
            source_row,
            source_day,
            source_day_order,
            canonical_label,
            binary_label,
            feature_hash_a,
            feature_hash_b
        FROM (
            SELECT
                h.source_file,
                h.source_row,
                h.source_day,
                h.source_day_order,
                h.canonical_label,
                h.binary_label,
                h.feature_hash_a,
                h.feature_hash_b,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        h.feature_hash_a,
                        h.feature_hash_b,
                        h.canonical_label
                    ORDER BY
                        h.source_day_order,
                        h.source_file,
                        h.source_row
                ) AS duplicate_rank
            FROM hashed h
            LEFT JOIN conflict_hashes c
                ON
                    h.feature_hash_a
                        = c.feature_hash_a
                    AND
                    h.feature_hash_b
                        = c.feature_hash_b
            WHERE
                c.feature_hash_a IS NULL
        )
        WHERE duplicate_rank = 1
        """
    )

    selected_rows = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM selected_rows
            """
        ).fetchone()[0]
    )

    duplicate_rows_removed = (
        nonconflict_rows
        - selected_rows
    )

    print(
        f"DUPLICATE_ROWS_REMOVED="
        f"{duplicate_rows_removed}"
    )

    print(
        f"DEDUPLICATED_ROWS="
        f"{selected_rows}"
    )

    print(
        "SPLIT_ASSIGNMENT_BEGIN"
    )

    connection.execute(
        f"""
        CREATE TABLE split_assignments AS
        WITH ranked AS (
            SELECT
                *,
                source_file
                    || ':'
                    || CAST(
                        source_row AS VARCHAR
                    )
                    AS row_id,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        binary_label
                    ORDER BY
                        md5(
                            source_file
                            || ':'
                            || CAST(
                                source_row AS VARCHAR
                            )
                            || '|{SPLIT_SEED}'
                        ),
                        source_file,
                        source_row
                ) AS split_rank,
                COUNT(*) OVER (
                    PARTITION BY
                        binary_label
                ) AS class_count
            FROM selected_rows
        )
        SELECT
            source_file,
            source_row,
            source_day,
            source_day_order,
            canonical_label,
            binary_label,
            feature_hash_a,
            feature_hash_b,
            row_id,
            CASE
                WHEN split_rank
                    <= CAST(
                        FLOOR(
                            class_count
                            * 0.70
                        )
                        AS BIGINT
                    )
                    THEN 'train'
                WHEN split_rank
                    <= CAST(
                        FLOOR(
                            class_count
                            * 0.85
                        )
                        AS BIGINT
                    )
                    THEN 'validation'
                ELSE 'test'
            END AS benchmark_split
        FROM ranked
        """
    )

    assignment_rows = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM split_assignments
            """
        ).fetchone()[0]
    )

    if assignment_rows != selected_rows:
        raise RuntimeError(
            "Split assignment row count mismatch"
        )

    feature_select = ",\n".join(
        f"h.{quote_identifier(column)}"
        for column in feature_columns
    )

    final_path = (
        PROCESSED
        / "cicids2017_clean.parquet"
    )

    assignment_path = (
        ARTIFACTS
        / "cicids_split_assignments.parquet"
    )

    if final_path.exists():
        final_path.unlink()

    if assignment_path.exists():
        assignment_path.unlink()

    final_sql_path = str(
        final_path
    ).replace(
        "'",
        "''"
    )

    assignment_sql_path = str(
        assignment_path
    ).replace(
        "'",
        "''"
    )

    print(
        "FINAL_PARQUET_WRITE_BEGIN"
    )

    connection.execute(
        f"""
        COPY (
            SELECT
                {feature_select},
                s.canonical_label,
                s.binary_label,
                s.source_file,
                s.source_day,
                s.source_day_order,
                s.source_row,
                s.feature_hash_a,
                s.feature_hash_b,
                s.row_id,
                s.benchmark_split
            FROM hashed h
            INNER JOIN split_assignments s
                ON
                    h.source_file
                        = s.source_file
                    AND
                    h.source_row
                        = s.source_row
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
            FROM split_assignments
        )
        TO '{assignment_sql_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    final_rows = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM split_assignments
            """
        ).fetchone()[0]
    )

    canonical_distribution = (
        connection.execute(
            """
            SELECT
                canonical_label,
                COUNT(*) AS count
            FROM split_assignments
            GROUP BY canonical_label
            ORDER BY
                count DESC,
                canonical_label
            """
        ).df()
    )

    binary_distribution = (
        connection.execute(
            """
            SELECT
                binary_label,
                COUNT(*) AS count
            FROM split_assignments
            GROUP BY binary_label
            ORDER BY binary_label
            """
        ).df()
    )

    split_distribution = (
        connection.execute(
            """
            SELECT
                benchmark_split,
                binary_label,
                COUNT(*) AS count
            FROM split_assignments
            GROUP BY
                benchmark_split,
                binary_label
            ORDER BY
                CASE benchmark_split
                    WHEN 'train'
                        THEN 0
                    WHEN 'validation'
                        THEN 1
                    ELSE 2
                END,
                binary_label
            """
        ).df()
    )

    day_distribution = (
        connection.execute(
            """
            SELECT
                source_day,
                binary_label,
                COUNT(*) AS count
            FROM split_assignments
            GROUP BY
                source_day,
                binary_label
            ORDER BY
                MIN(source_day_order),
                binary_label
            """
        ).df()
    )

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

    input_rows = read_input_row_count()

    if input_rows is None:
        input_rows = staged_rows

    invalid_rows = (
        input_rows
        - staged_rows
    )

    report = {
        "dataset": "CICIDS2017",
        "protocol_version": 1,
        "feature_count": len(
            feature_columns
        ),
        "input_rows": input_rows,
        "invalid_rows_dropped": invalid_rows,
        "valid_rows_staged": staged_rows,
        "fingerprint": {
            "representation": (
                "two 64-bit pandas row hashes"
            ),
            "effective_bits": 128,
            "hash_a": (
                "feature columns in canonical order"
            ),
            "hash_b": (
                "feature columns in reverse order"
            ),
            "purpose": (
                "memory-bounded duplicate "
                "and label-conflict detection"
            )
        },
        "conflicting_feature_groups": (
            conflict_feature_groups
        ),
        "conflicting_rows_dropped": (
            conflict_rows
        ),
        "duplicate_rows_dropped": (
            duplicate_rows_removed
        ),
        "final_rows": final_rows,
        "binary_definition": {
            "0": "BENIGN",
            "1": "ATTACK"
        },
        "benchmark_split": {
            "seed": SPLIT_SEED,
            "method": (
                "Exact class-stratified "
                "70/15/15 split using "
                "deterministic MD5 ordering "
                "of stable source row identifiers"
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
            "The raw CSV contains U+FFFD "
            "literally in the three Web Attack "
            "labels. Those labels were mapped "
            "deterministically to canonical "
            "ASCII class names during staging."
        ),
        "normalization_policy": (
            "No scaling was performed during "
            "preprocessing. All later scalers "
            "must be fit on training data only."
        ),
        "imbalance_policy": (
            "No class downsampling was performed "
            "during preprocessing."
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

    write_day_protocol()

    connection.close()

    print()
    print(
        "CICIDS_FINALIZATION_COMPLETE"
    )

    print(
        f"INPUT_ROWS={input_rows}"
    )

    print(
        f"INVALID_ROWS_DROPPED="
        f"{invalid_rows}"
    )

    print(
        f"VALID_ROWS_STAGED="
        f"{staged_rows}"
    )

    print(
        f"CONFLICTING_FEATURE_GROUPS="
        f"{conflict_feature_groups}"
    )

    print(
        f"CONFLICTING_ROWS_DROPPED="
        f"{conflict_rows}"
    )

    print(
        f"DUPLICATE_ROWS_DROPPED="
        f"{duplicate_rows_removed}"
    )

    print(
        f"FINAL_ROWS={final_rows}"
    )

    print(
        f"FEATURES="
        f"{len(feature_columns)}"
    )

    print(
        "BINARY_TARGET=BENIGN_VS_ATTACK"
    )

    print(
        "BENCHMARK_SPLIT="
        "70_15_15_EXACT_CLASS_STRATIFIED"
    )

    print(
        "DAY_HOLDOUT_PROTOCOL=PASS"
    )

    print(
        f"OUTPUT={final_path}"
    )


if __name__ == "__main__":
    main()