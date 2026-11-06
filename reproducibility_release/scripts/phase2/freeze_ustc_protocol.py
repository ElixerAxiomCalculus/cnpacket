import csv
import hashlib
import json
import struct
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
FLOWS = ROOT / "data" / "processed" / "ustc-tfc2016" / "flows"
ARTIFACTS = ROOT / "artifacts" / "phase2"

MIN_PACKETS = 3
SEED = "20261011"
FEWSHOT_K = [1, 5, 10]


def representation_fingerprint(
    sizes,
    interarrivals
):
    if len(sizes) != len(interarrivals):
        raise RuntimeError(
            "Sequence length mismatch"
        )

    digest = hashlib.sha256()

    digest.update(
        struct.pack(
            ">H",
            len(sizes)
        )
    )

    for value in sizes:
        digest.update(
            struct.pack(
                ">i",
                int(value)
            )
        )

    for value in interarrivals:
        microseconds = int(
            round(
                float(value)
                * 1_000_000
            )
        )

        digest.update(
            struct.pack(
                ">q",
                microseconds
            )
        )

    return digest.hexdigest()


def stable_support_key(
    family,
    flow_id
):
    value = (
        f"{SEED}|"
        f"{family}|"
        f"{flow_id}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def group_partition(
    family_frame,
    thresholds,
    labels
):
    groups = (
        family_frame
        .groupby(
            "representation_fingerprint",
            as_index=False
        )
        .agg(
            group_start=(
                "start_time",
                "min"
            ),
            group_size=(
                "flow_id",
                "size"
            )
        )
        .sort_values(
            [
                "group_start",
                "representation_fingerprint"
            ],
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )

    total = int(
        groups["group_size"].sum()
    )

    assignments = {}
    cumulative = 0

    for row in groups.itertuples(
        index=False
    ):
        midpoint = (
            cumulative
            + row.group_size / 2.0
        ) / total

        selected = labels[-1]

        for threshold, label in zip(
            thresholds,
            labels
        ):
            if midpoint <= threshold:
                selected = label
                break

        assignments[
            row.representation_fingerprint
        ] = selected

        cumulative += int(
            row.group_size
        )

    return assignments


def main():
    files = sorted(
        FLOWS.glob(
            "Malware__*.parquet"
        )
    )

    if len(files) != 10:
        raise RuntimeError(
            f"Expected 10 malware Parquets, found {len(files)}"
        )

    metadata_frames = []

    for path in files:
        table = pq.read_table(
            path,
            columns=[
                "class_name",
                "capture",
                "flow_id",
                "start_time",
                "packet_count",
                "sequence_packets",
                "signed_packet_sizes",
                "interarrival_seconds"
            ]
        )

        frame = table.to_pandas()

        frame = frame[
            frame["packet_count"]
            >= MIN_PACKETS
        ].copy()

        fingerprints = []

        for row in frame.itertuples(
            index=False
        ):
            sizes = list(
                row.signed_packet_sizes
            )

            interarrivals = list(
                row.interarrival_seconds
            )

            if (
                len(sizes)
                != row.sequence_packets
            ):
                raise RuntimeError(
                    f"Sequence packet mismatch "
                    f"for {row.flow_id}"
                )

            fingerprints.append(
                representation_fingerprint(
                    sizes,
                    interarrivals
                )
            )

        frame[
            "representation_fingerprint"
        ] = fingerprints

        metadata_frames.append(
            frame[
                [
                    "class_name",
                    "capture",
                    "flow_id",
                    "start_time",
                    "packet_count",
                    "sequence_packets",
                    "representation_fingerprint"
                ]
            ].copy()
        )

        print(
            f"LOADED={path.name} "
            f"VALID_GE3={len(frame)}"
        )

    manifest = pd.concat(
        metadata_frames,
        ignore_index=True
    )

    if manifest["flow_id"].duplicated().any():
        duplicates = manifest[
            manifest[
                "flow_id"
            ].duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            "Duplicate flow_id values detected: "
            f"{len(duplicates)}"
        )

    initial_rows = len(
        manifest
    )

    family_counts_per_representation = (
        manifest
        .groupby(
            "representation_fingerprint"
        )[
            "class_name"
        ]
        .nunique()
    )

    conflicting_fingerprints = set(
        family_counts_per_representation[
            family_counts_per_representation
            > 1
        ].index
    )

    manifest[
        "cross_family_conflict"
    ] = (
        manifest[
            "representation_fingerprint"
        ]
        .isin(
            conflicting_fingerprints
        )
    )

    conflict_rows = int(
        manifest[
            "cross_family_conflict"
        ].sum()
    )

    usable = manifest[
        ~manifest[
            "cross_family_conflict"
        ]
    ].copy()

    families = sorted(
        usable[
            "class_name"
        ].unique()
    )

    if len(families) != 10:
        raise RuntimeError(
            f"Expected 10 usable families, found {families}"
        )

    closed_split = {}
    adaptation_partition = {}

    for family in families:
        family_frame = usable[
            usable["class_name"]
            == family
        ].copy()

        closed_assignments = (
            group_partition(
                family_frame,
                thresholds=[
                    0.60,
                    0.80,
                    1.00
                ],
                labels=[
                    "train",
                    "validation",
                    "test"
                ]
            )
        )

        adaptation_assignments = (
            group_partition(
                family_frame,
                thresholds=[
                    0.20,
                    0.40,
                    1.00
                ],
                labels=[
                    "support_pool",
                    "calibration_pool",
                    "query_pool"
                ]
            )
        )

        for fingerprint, value in (
            closed_assignments.items()
        ):
            closed_split[
                (
                    family,
                    fingerprint
                )
            ] = value

        for fingerprint, value in (
            adaptation_assignments.items()
        ):
            adaptation_partition[
                (
                    family,
                    fingerprint
                )
            ] = value

    usable[
        "closed_split"
    ] = [
        closed_split[
            (
                row.class_name,
                row.representation_fingerprint
            )
        ]
        for row in usable.itertuples(
            index=False
        )
    ]

    usable[
        "adaptation_partition"
    ] = [
        adaptation_partition[
            (
                row.class_name,
                row.representation_fingerprint
            )
        ]
        for row in usable.itertuples(
            index=False
        )
    ]

    manifest_path = (
        ARTIFACTS
        / "ustc_malware_protocol_manifest.parquet"
    )

    pq.write_table(
        pa.Table.from_pandas(
            usable,
            preserve_index=False
        ),
        manifest_path,
        compression="zstd"
    )

    summary_rows = []

    for family in families:
        family_frame = usable[
            usable[
                "class_name"
            ]
            == family
        ]

        original_family_rows = (
            manifest[
                manifest[
                    "class_name"
                ]
                == family
            ]
        )

        summary_rows.append(
            {
                "family": family,
                "flows_ge3_before_conflict_filter": (
                    len(
                        original_family_rows
                    )
                ),
                "cross_family_conflict_rows_removed": (
                    int(
                        original_family_rows[
                            "cross_family_conflict"
                        ].sum()
                    )
                ),
                "usable_flows": (
                    len(
                        family_frame
                    )
                ),
                "representation_groups": (
                    family_frame[
                        "representation_fingerprint"
                    ].nunique()
                ),
                "duplicate_representation_rows": (
                    len(
                        family_frame
                    )
                    - family_frame[
                        "representation_fingerprint"
                    ].nunique()
                ),
                "closed_train": (
                    int(
                        (
                            family_frame[
                                "closed_split"
                            ]
                            == "train"
                        ).sum()
                    )
                ),
                "closed_validation": (
                    int(
                        (
                            family_frame[
                                "closed_split"
                            ]
                            == "validation"
                        ).sum()
                    )
                ),
                "closed_test": (
                    int(
                        (
                            family_frame[
                                "closed_split"
                            ]
                            == "test"
                        ).sum()
                    )
                ),
                "support_pool": (
                    int(
                        (
                            family_frame[
                                "adaptation_partition"
                            ]
                            == "support_pool"
                        ).sum()
                    )
                ),
                "calibration_pool": (
                    int(
                        (
                            family_frame[
                                "adaptation_partition"
                            ]
                            == "calibration_pool"
                        ).sum()
                    )
                ),
                "query_pool": (
                    int(
                        (
                            family_frame[
                                "adaptation_partition"
                            ]
                            == "query_pool"
                        ).sum()
                    )
                )
            }
        )

    summary_path = (
        ARTIFACTS
        / "ustc_malware_protocol_summary.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False
    )

    runs = []

    support_rows = []

    for run_index, target_family in enumerate(
        families
    ):
        calibration_family = families[
            (
                run_index + 1
            )
            % len(families)
        ]

        known_families = [
            family
            for family in families
            if family
            not in {
                target_family,
                calibration_family
            }
        ]

        target_frame = usable[
            usable[
                "class_name"
            ]
            == target_family
        ]

        support_pool = target_frame[
            target_frame[
                "adaptation_partition"
            ]
            == "support_pool"
        ].copy()

        query_pool = target_frame[
            target_frame[
                "adaptation_partition"
            ]
            == "query_pool"
        ].copy()

        if len(support_pool) < max(
            FEWSHOT_K
        ):
            raise RuntimeError(
                f"{target_family} has only "
                f"{len(support_pool)} support flows"
            )

        if len(query_pool) < 50:
            raise RuntimeError(
                f"{target_family} has only "
                f"{len(query_pool)} query flows"
            )

        support_pool[
            "support_key"
        ] = [
            stable_support_key(
                target_family,
                flow_id
            )
            for flow_id
            in support_pool[
                "flow_id"
            ]
        ]

        support_pool = (
            support_pool
            .sort_values(
                [
                    "support_key",
                    "flow_id"
                ],
                kind="mergesort"
            )
            .reset_index(
                drop=True
            )
        )

        selected_ten = support_pool.head(
            max(
                FEWSHOT_K
            )
        )

        for k in FEWSHOT_K:
            selected = selected_ten.head(
                k
            )

            for support_rank, row in enumerate(
                selected.itertuples(
                    index=False
                ),
                start=1
            ):
                support_rows.append(
                    {
                        "run": run_index,
                        "target_family": (
                            target_family
                        ),
                        "k": k,
                        "support_rank": (
                            support_rank
                        ),
                        "flow_id": row.flow_id
                    }
                )

        runs.append(
            {
                "run": run_index,
                "target_unknown_family": (
                    target_family
                ),
                "calibration_unknown_family": (
                    calibration_family
                ),
                "known_families": (
                    known_families
                ),
                "known_training_partition": (
                    "closed_split=train"
                ),
                "known_validation_partition": (
                    "closed_split=validation"
                ),
                "known_test_partition": (
                    "closed_split=test"
                ),
                "unknown_threshold_calibration": (
                    "calibration family "
                    "adaptation_partition="
                    "calibration_pool"
                ),
                "zero_shot_unknown_test": (
                    "target family "
                    "adaptation_partition="
                    "query_pool"
                ),
                "few_shot_support": (
                    "target family "
                    "adaptation_partition="
                    "support_pool"
                ),
                "few_shot_query": (
                    "target family "
                    "adaptation_partition="
                    "query_pool"
                ),
                "few_shot_k": FEWSHOT_K
            }
        )

    support_path = (
        ARTIFACTS
        / "ustc_fewshot_support.csv"
    )

    with support_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "target_family",
                "k",
                "support_rank",
                "flow_id"
            ]
        )

        writer.writeheader()
        writer.writerows(
            support_rows
        )

    protocol = {
        "dataset": "USTC-TFC2016",
        "protocol_version": 1,
        "task_scope": (
            "malware-family classification only"
        ),
        "families": families,
        "minimum_packets_per_flow": (
            MIN_PACKETS
        ),
        "maximum_stored_packets": 50,
        "model_input": [
            "signed_packet_size",
            "interarrival_time",
            "sequence_mask"
        ],
        "payload_bytes_used": False,
        "excluded_from_primary_ustc_task": {
            "benign_application_classification": (
                "Several benign classes have "
                "zero or near-zero flows with "
                "at least three packets"
            )
        },
        "representation_grouping": {
            "fingerprint": (
                "SHA-256 over first up-to-50 signed "
                "packet sizes and IAT values rounded "
                "to microseconds"
            ),
            "cross_family_conflicts": (
                "excluded from all experiments"
            ),
            "within_family_duplicates": (
                "kept in the same split"
            )
        },
        "closed_set": {
            "families": 10,
            "split_method": (
                "representation-group-aware "
                "temporal 60/20/20"
            ),
            "training_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "training_imbalance_policy": (
                "retain all unique samples and "
                "use class-weighted loss or "
                "balanced sampling inside "
                "training only"
            )
        },
        "open_set": {
            "runs": len(
                runs
            ),
            "design": (
                "one target unknown family plus "
                "one separate calibration-unknown "
                "family per run; remaining eight "
                "families are known"
            ),
            "threshold_policy": (
                "unknown rejection threshold may "
                "use known validation samples and "
                "the calibration-unknown family, "
                "never the target unknown family"
            ),
            "runs_detail": runs
        },
        "few_shot": {
            "k": FEWSHOT_K,
            "support_selection": (
                "nested deterministic support sets "
                "from temporally early support_pool"
            ),
            "query": (
                "temporally later query_pool"
            )
        },
        "early_classification": {
            "ustc_primary": False,
            "reason": (
                "Tinba has insufficient flows for "
                "uniform N=5/10/20/50 evaluation"
            ),
            "planned_dataset": "ISCXVPN2016"
        },
        "known_limitation": (
            "Most USTC classes originate from one "
            "capture, so closed-set separation is "
            "flow/representation-group temporal "
            "separation rather than independent "
            "capture separation."
        )
    }

    protocol_path = (
        ARTIFACTS
        / "ustc_protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2
        )
        + "\n"
    )

    closed_leakage = (
        usable
        .groupby(
            [
                "class_name",
                "representation_fingerprint"
            ]
        )[
            "closed_split"
        ]
        .nunique()
        .max()
    )

    adaptation_leakage = (
        usable
        .groupby(
            [
                "class_name",
                "representation_fingerprint"
            ]
        )[
            "adaptation_partition"
        ]
        .nunique()
        .max()
    )

    if closed_leakage != 1:
        raise RuntimeError(
            "Closed-set representation leakage"
        )

    if adaptation_leakage != 1:
        raise RuntimeError(
            "Adaptation representation leakage"
        )

    minimum_usable = int(
        usable
        .groupby(
            "class_name"
        )
        .size()
        .min()
    )

    print()
    print(
        f"INITIAL_MALWARE_FLOWS_GE3="
        f"{initial_rows}"
    )

    print(
        f"CROSS_FAMILY_CONFLICT_GROUPS="
        f"{len(conflicting_fingerprints)}"
    )

    print(
        f"CROSS_FAMILY_CONFLICT_ROWS_DROPPED="
        f"{conflict_rows}"
    )

    print(
        f"USABLE_MALWARE_FLOWS="
        f"{len(usable)}"
    )

    print(
        f"MALWARE_FAMILIES="
        f"{len(families)}"
    )

    print(
        f"MIN_USABLE_FAMILY_FLOWS="
        f"{minimum_usable}"
    )

    print(
        "CLOSED_SET_FAMILIES=10"
    )

    print(
        "CLOSED_SET_SPLIT="
        "60_20_20_TEMPORAL_GROUPED"
    )

    print(
        f"OPEN_SET_RUNS="
        f"{len(runs)}"
    )

    print(
        "FEWSHOT_K=1,5,10"
    )

    print(
        "REPRESENTATION_SPLIT_LEAKAGE=0"
    )

    print(
        "TARGET_UNKNOWN_THRESHOLD_LEAKAGE=0"
    )

    print(
        "USTC_PROTOCOL=PASS"
    )

    print(
        f"MANIFEST={manifest_path}"
    )

    print(
        f"SUMMARY={summary_path}"
    )

    print(
        f"PROTOCOL={protocol_path}"
    )

    print(
        f"FEWSHOT_SUPPORT={support_path}"
    )


if __name__ == "__main__":
    main()