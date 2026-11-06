import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
FLOW_DIRECTORY = ROOT / "data" / "processed" / "iscxvpn2016" / "flows"
ARTIFACTS = ROOT / "artifacts" / "phase2"
CAPTURE_SUMMARY = ARTIFACTS / "iscx_flow_capture_summary.csv"

MODEL_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "iscxvpn2016"
    / "iscx_model_ready.parquet"
)

MODEL_MANIFEST = (
    ARTIFACTS
    / "iscx_model_manifest.parquet"
)

PROTOCOL_PATH = (
    ARTIFACTS
    / "iscx_model_protocol.json"
)

CV_SUMMARY_PATH = (
    ARTIFACTS
    / "iscx_model_cv_summary.csv"
)

STRATUM_SUMMARY_PATH = (
    ARTIFACTS
    / "iscx_model_stratum_summary.csv"
)

CAP_EFFECT_PATH = (
    ARTIFACTS
    / "iscx_training_cap_effect.csv"
)

SEED = "20261011"
PRIMARY_N = 5
EARLY_LENGTHS = [5, 10, 20, 50]
COMMON_EARLY_MINIMUM = 50
TRAINING_CAPTURE_CAP = 500

ACTIVITY_LABELS = {
    "chat": 0,
    "file_transfer": 1,
    "streaming": 2,
    "voip": 3
}

DOMAIN_LABELS = {
    "nonvpn": 0,
    "vpn": 1
}


def stable_training_key(
    archive,
    capture,
    flow_id
):
    value = (
        f"{SEED}|"
        f"{archive}|"
        f"{capture}|"
        f"{flow_id}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def role_for_fold(
    fold,
    run
):
    test_fold = run
    validation_fold = (
        run + 1
    ) % 5

    if fold == test_fold:
        return "test"

    if fold == validation_fold:
        return "validation"

    return "train"


def capped_training_frame(
    frame
):
    ordered = frame.sort_values(
        [
            "capture_id",
            "training_key",
            "flow_id"
        ],
        kind="mergesort"
    )

    return (
        ordered
        .groupby(
            "capture_id",
            sort=False,
            group_keys=False
        )
        .head(
            TRAINING_CAPTURE_CAP
        )
        .copy()
    )


def main():
    flow_files = sorted(
        FLOW_DIRECTORY.glob(
            "*.parquet"
        )
    )

    if len(flow_files) != 133:
        raise RuntimeError(
            f"Expected 133 flow Parquets, "
            f"found {len(flow_files)}"
        )

    capture_summary = pd.read_csv(
        CAPTURE_SUMMARY
    )

    if len(capture_summary) != 133:
        raise RuntimeError(
            f"Expected 133 capture-summary rows, "
            f"found {len(capture_summary)}"
        )

    expected_totals = {
        5: int(
            capture_summary[
                "flows_ge5"
            ].sum()
        ),
        10: int(
            capture_summary[
                "flows_ge10"
            ].sum()
        ),
        20: int(
            capture_summary[
                "flows_ge20"
            ].sum()
        ),
        50: int(
            capture_summary[
                "flows_ge50"
            ].sum()
        )
    }

    dataset = ds.dataset(
        str(
            FLOW_DIRECTORY
        ),
        format="parquet"
    )

    table = dataset.to_table(
        filter=(
            ds.field(
                "packet_count"
            )
            >= PRIMARY_N
        ),
        columns=[
            "domain",
            "activity",
            "fold",
            "archive",
            "capture",
            "flow_id",
            "packet_count",
            "sequence_packets",
            "signed_packet_sizes",
            "interarrival_seconds"
        ]
    )

    frame = table.to_pandas()

    if len(frame) != expected_totals[5]:
        raise RuntimeError(
            f"N=5 count mismatch: "
            f"{len(frame)} != "
            f"{expected_totals[5]}"
        )

    if frame[
        "flow_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate flow IDs detected"
        )

    observed_activities = set(
        frame[
            "activity"
        ].unique()
    )

    if observed_activities != set(
        ACTIVITY_LABELS
    ):
        raise RuntimeError(
            f"Unexpected activities: "
            f"{observed_activities}"
        )

    observed_domains = set(
        frame[
            "domain"
        ].unique()
    )

    if observed_domains != set(
        DOMAIN_LABELS
    ):
        raise RuntimeError(
            f"Unexpected domains: "
            f"{observed_domains}"
        )

    observed_folds = sorted(
        frame[
            "fold"
        ].unique().tolist()
    )

    if observed_folds != [
        0,
        1,
        2,
        3,
        4
    ]:
        raise RuntimeError(
            f"Unexpected folds: "
            f"{observed_folds}"
        )

    frame[
        "signed_packet_sizes"
    ] = frame[
        "signed_packet_sizes"
    ].map(list)

    frame[
        "interarrival_seconds"
    ] = frame[
        "interarrival_seconds"
    ].map(list)

    bad_sequence_rows = 0

    for row in frame.itertuples(
        index=False
    ):
        expected_sequence_packets = min(
            int(
                row.packet_count
            ),
            50
        )

        if (
            int(
                row.sequence_packets
            )
            != expected_sequence_packets
        ):
            bad_sequence_rows += 1
            continue

        if (
            len(
                row.signed_packet_sizes
            )
            != expected_sequence_packets
        ):
            bad_sequence_rows += 1
            continue

        if (
            len(
                row.interarrival_seconds
            )
            != expected_sequence_packets
        ):
            bad_sequence_rows += 1

    if bad_sequence_rows != 0:
        raise RuntimeError(
            f"Invalid sequence rows: "
            f"{bad_sequence_rows}"
        )

    frame[
        "activity_label"
    ] = frame[
        "activity"
    ].map(
        ACTIVITY_LABELS
    ).astype(
        "int8"
    )

    frame[
        "domain_label"
    ] = frame[
        "domain"
    ].map(
        DOMAIN_LABELS
    ).astype(
        "int8"
    )

    frame[
        "capture_id"
    ] = (
        frame[
            "archive"
        ]
        + "::"
        + frame[
            "capture"
        ]
    )

    frame[
        "training_key"
    ] = [
        stable_training_key(
            row.archive,
            row.capture,
            row.flow_id
        )
        for row in frame.itertuples(
            index=False
        )
    ]

    for n in EARLY_LENGTHS:
        actual = int(
            (
                frame[
                    "packet_count"
                ]
                >= n
            ).sum()
        )

        if actual != expected_totals[n]:
            raise RuntimeError(
                f"N={n} count mismatch: "
                f"{actual} != "
                f"{expected_totals[n]}"
            )

    frame = frame.sort_values(
        [
            "fold",
            "domain",
            "activity",
            "capture_id",
            "training_key",
            "flow_id"
        ],
        kind="mergesort"
    ).reset_index(
        drop=True
    )

    model_columns = [
        "flow_id",
        "domain",
        "domain_label",
        "activity",
        "activity_label",
        "fold",
        "archive",
        "capture",
        "capture_id",
        "packet_count",
        "sequence_packets",
        "training_key",
        "signed_packet_sizes",
        "interarrival_seconds"
    ]

    model_frame = frame[
        model_columns
    ].copy()

    pq.write_table(
        pa.Table.from_pandas(
            model_frame,
            preserve_index=False
        ),
        MODEL_DATASET,
        compression="zstd"
    )

    manifest_columns = [
        "flow_id",
        "domain",
        "domain_label",
        "activity",
        "activity_label",
        "fold",
        "archive",
        "capture",
        "capture_id",
        "packet_count",
        "sequence_packets",
        "training_key"
    ]

    pq.write_table(
        pa.Table.from_pandas(
            frame[
                manifest_columns
            ],
            preserve_index=False
        ),
        MODEL_MANIFEST,
        compression="zstd"
    )

    primary_grid = (
        pd.MultiIndex.from_product(
            [
                range(5),
                sorted(
                    DOMAIN_LABELS
                ),
                sorted(
                    ACTIVITY_LABELS
                )
            ],
            names=[
                "fold",
                "domain",
                "activity"
            ]
        )
        .to_frame(
            index=False
        )
    )

    primary_counts = (
        frame
        .groupby(
            [
                "fold",
                "domain",
                "activity"
            ]
        )
        .size()
        .rename(
            "count"
        )
        .reset_index()
    )

    primary_grid = (
        primary_grid
        .merge(
            primary_counts,
            on=[
                "fold",
                "domain",
                "activity"
            ],
            how="left"
        )
        .fillna(
            {
                "count": 0
            }
        )
    )

    primary_grid[
        "count"
    ] = primary_grid[
        "count"
    ].astype(
        int
    )

    minimum_primary_stratum = int(
        primary_grid[
            "count"
        ].min()
    )

    if minimum_primary_stratum < 5:
        raise RuntimeError(
            f"Primary N=5 fold/domain/activity "
            f"minimum is only "
            f"{minimum_primary_stratum}"
        )

    early_frame = frame[
        frame[
            "packet_count"
        ]
        >= COMMON_EARLY_MINIMUM
    ].copy()

    early_grid = (
        pd.MultiIndex.from_product(
            [
                range(5),
                sorted(
                    ACTIVITY_LABELS
                )
            ],
            names=[
                "fold",
                "activity"
            ]
        )
        .to_frame(
            index=False
        )
    )

    early_counts = (
        early_frame
        .groupby(
            [
                "fold",
                "activity"
            ]
        )
        .size()
        .rename(
            "count"
        )
        .reset_index()
    )

    early_grid = (
        early_grid
        .merge(
            early_counts,
            on=[
                "fold",
                "activity"
            ],
            how="left"
        )
        .fillna(
            {
                "count": 0
            }
        )
    )

    early_grid[
        "count"
    ] = early_grid[
        "count"
    ].astype(
        int
    )

    minimum_early_test_class = int(
        early_grid[
            "count"
        ].min()
    )

    if minimum_early_test_class < 10:
        raise RuntimeError(
            f"N=50 pooled fold/activity "
            f"minimum is only "
            f"{minimum_early_test_class}"
        )

    cv_summary_rows = []
    stratum_rows = []
    cap_effect_rows = []
    cv_runs = []

    for run in range(5):
        test_fold = run
        validation_fold = (
            run + 1
        ) % 5

        training_folds = [
            fold
            for fold in range(5)
            if fold
            not in {
                test_fold,
                validation_fold
            }
        ]

        cv_runs.append(
            {
                "run": run,
                "training_folds": (
                    training_folds
                ),
                "validation_fold": (
                    validation_fold
                ),
                "test_fold": (
                    test_fold
                )
            }
        )

        working = frame.copy()

        working[
            "role"
        ] = working[
            "fold"
        ].map(
            lambda fold: role_for_fold(
                int(
                    fold
                ),
                run
            )
        )

        capture_role_max = (
            working
            .groupby(
                "capture_id"
            )[
                "role"
            ]
            .nunique()
            .max()
        )

        if capture_role_max != 1:
            raise RuntimeError(
                f"Capture leakage in run "
                f"{run}"
            )

        for n in EARLY_LENGTHS:
            eligible = working[
                working[
                    "packet_count"
                ]
                >= n
            ].copy()

            for role in [
                "train",
                "validation",
                "test"
            ]:
                role_frame = eligible[
                    eligible[
                        "role"
                    ]
                    == role
                ].copy()

                raw_count = len(
                    role_frame
                )

                if role == "train":
                    effective = (
                        capped_training_frame(
                            role_frame
                        )
                    )
                else:
                    effective = role_frame

                effective_count = len(
                    effective
                )

                cv_summary_rows.append(
                    {
                        "run": run,
                        "n": n,
                        "role": role,
                        "raw_count": (
                            raw_count
                        ),
                        "effective_count": (
                            effective_count
                        )
                    }
                )

                for domain in sorted(
                    DOMAIN_LABELS
                ):
                    for activity in sorted(
                        ACTIVITY_LABELS
                    ):
                        raw_stratum = (
                            role_frame[
                                (
                                    role_frame[
                                        "domain"
                                    ]
                                    == domain
                                )
                                &
                                (
                                    role_frame[
                                        "activity"
                                    ]
                                    == activity
                                )
                            ]
                        )

                        effective_stratum = (
                            effective[
                                (
                                    effective[
                                        "domain"
                                    ]
                                    == domain
                                )
                                &
                                (
                                    effective[
                                        "activity"
                                    ]
                                    == activity
                                )
                            ]
                        )

                        stratum_rows.append(
                            {
                                "run": run,
                                "n": n,
                                "role": role,
                                "domain": (
                                    domain
                                ),
                                "activity": (
                                    activity
                                ),
                                "raw_count": (
                                    len(
                                        raw_stratum
                                    )
                                ),
                                "effective_count": (
                                    len(
                                        effective_stratum
                                    )
                                )
                            }
                        )

            training = eligible[
                eligible[
                    "role"
                ]
                == "train"
            ].copy()

            capped = capped_training_frame(
                training
            )

            cap_effect_rows.append(
                {
                    "run": run,
                    "n": n,
                    "training_before_cap": (
                        len(
                            training
                        )
                    ),
                    "training_after_cap": (
                        len(
                            capped
                        )
                    ),
                    "flows_removed": (
                        len(
                            training
                        )
                        - len(
                            capped
                        )
                    ),
                    "captures_over_cap": (
                        int(
                            (
                                training
                                .groupby(
                                    "capture_id"
                                )
                                .size()
                                > TRAINING_CAPTURE_CAP
                            ).sum()
                        )
                    )
                }
            )

    pd.DataFrame(
        cv_summary_rows
    ).to_csv(
        CV_SUMMARY_PATH,
        index=False
    )

    pd.DataFrame(
        stratum_rows
    ).to_csv(
        STRATUM_SUMMARY_PATH,
        index=False
    )

    pd.DataFrame(
        cap_effect_rows
    ).to_csv(
        CAP_EFFECT_PATH,
        index=False
    )

    protocol = {
        "dataset": "ISCXVPN2016",
        "model_protocol_version": 1,
        "seed": SEED,
        "target": "activity",
        "activity_labels": (
            ACTIVITY_LABELS
        ),
        "domain_attribute": (
            "vpn_status"
        ),
        "domain_labels": (
            DOMAIN_LABELS
        ),
        "capture_count": 133,
        "activities": [
            "chat",
            "file_transfer",
            "streaming",
            "voip"
        ],
        "input_representation": {
            "features": [
                "signed_packet_size",
                "interarrival_time",
                "sequence_mask"
            ],
            "payload_bytes_used": False,
            "endpoint_addresses_used": False,
            "endpoint_ports_used": False,
            "protocol_identifier_used": False,
            "maximum_stored_packets": 50,
            "direction": (
                "positive for first-observed "
                "flow direction and negative "
                "for reverse direction"
            )
        },
        "primary_experiment": {
            "sequence_length": 5,
            "eligible_flows": (
                expected_totals[5]
            ),
            "design": (
                "four-class activity "
                "classification using the "
                "first five packets"
            ),
            "evaluation": (
                "five capture-group folds"
            ),
            "reporting": [
                "overall metrics",
                "per-class metrics",
                "nonvpn test-subset metrics",
                "vpn test-subset metrics"
            ]
        },
        "cross_validation": {
            "fold_count": 5,
            "runs": cv_runs,
            "assignment": (
                "run r uses fold r for test, "
                "fold (r+1) mod 5 for "
                "validation, and the other "
                "three folds for training"
            ),
            "grouping_unit": (
                "capture_file"
            ),
            "capture_leakage": 0
        },
        "training_capture_cap": {
            "maximum_flows_per_capture": (
                TRAINING_CAPTURE_CAP
            ),
            "applies_to": (
                "training partitions only"
            ),
            "selection": (
                "deterministic SHA-256 rank "
                "using seed, archive, capture "
                "and flow_id after sequence-"
                "length eligibility filtering"
            ),
            "validation_capped": False,
            "test_capped": False
        },
        "early_classification": {
            "sequence_lengths": (
                EARLY_LENGTHS
            ),
            "common_cohort_requirement": (
                "packet_count >= 50"
            ),
            "common_cohort_flows": (
                expected_totals[50]
            ),
            "comparison": (
                "the same flows are evaluated "
                "using their first 5, 10, 20 "
                "and 50 packets"
            ),
            "primary_reporting_scope": (
                "pooled-domain four-class "
                "activity classification"
            ),
            "domain_specific_results": (
                "secondary because several "
                "fold/domain/activity cells "
                "are small at N=50"
            )
        },
        "normalization": {
            "policy": (
                "fit any data-dependent "
                "normalization using training "
                "data only"
            ),
            "padding": (
                "zero padding with an explicit "
                "sequence mask"
            )
        },
        "imbalance": {
            "preprocessing_downsampling": False,
            "allowed_training_methods": [
                "class-weighted loss",
                "balanced training sampler"
            ],
            "validation_resampling": False,
            "test_resampling": False
        },
        "eligibility_totals": {
            str(
                n
            ): expected_totals[n]
            for n in EARLY_LENGTHS
        },
        "minimum_primary_fold_domain_activity_count": (
            minimum_primary_stratum
        ),
        "minimum_n50_fold_pooled_activity_count": (
            minimum_early_test_class
        ),
        "known_limitations": [
            (
                "The N=50 common cohort is "
                "substantially smaller than "
                "the primary N=5 cohort."
            ),
            (
                "Capture sizes and flow counts "
                "are highly imbalanced; the "
                "training-only capture cap "
                "prevents a few recordings "
                "from dominating optimization."
            )
        ]
    }

    PROTOCOL_PATH.write_text(
        json.dumps(
            protocol,
            indent=2
        )
        + "\n"
    )

    print(
        f"MODEL_ROWS_N5="
        f"{len(frame)}"
    )

    for n in EARLY_LENGTHS:
        print(
            f"FLOWS_GE{n}="
            f"{expected_totals[n]}"
        )

    print(
        f"MIN_PRIMARY_FOLD_DOMAIN_ACTIVITY="
        f"{minimum_primary_stratum}"
    )

    print(
        f"MIN_N50_FOLD_POOLED_ACTIVITY="
        f"{minimum_early_test_class}"
    )

    print(
        f"TRAINING_CAPTURE_CAP="
        f"{TRAINING_CAPTURE_CAP}"
    )

    print(
        "VALIDATION_TEST_CAPPED=FALSE"
    )

    print(
        "CAPTURE_SPLIT_LEAKAGE=0"
    )

    print(
        "PRIMARY_SEQUENCE_LENGTH=5"
    )

    print(
        "EARLY_SEQUENCE_LENGTHS=5,10,20,50"
    )

    print(
        "EARLY_COMPARISON_COHORT=GE50"
    )

    print(
        "ISCX_MODEL_PROTOCOL=PASS"
    )

    print(
        f"MODEL_DATASET={MODEL_DATASET}"
    )

    print(
        f"MODEL_MANIFEST={MODEL_MANIFEST}"
    )

    print(
        f"PROTOCOL={PROTOCOL_PATH}"
    )


if __name__ == "__main__":
    main()