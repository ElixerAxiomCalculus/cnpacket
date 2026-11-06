from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FLOW_DIRECTORY = (
    ROOT
    / "data"
    / "processed"
    / "ustc-tfc2016"
    / "flows"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "ustc_malware_protocol_manifest.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
    / "ustc_closed_model_ready.parquet"
)

FAMILIES = [
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


def main():
    manifest = pd.read_parquet(
        MANIFEST_PATH
    )

    if len(manifest) != 229581:
        raise RuntimeError(
            f"Unexpected manifest rows: {len(manifest)}"
        )

    if manifest[
        "cross_family_conflict"
    ].any():
        raise RuntimeError(
            "Conflict rows remain in frozen manifest"
        )

    if manifest.duplicated(
        [
            "class_name",
            "flow_id"
        ]
    ).any():
        raise RuntimeError(
            "Duplicate manifest keys"
        )

    frames = []

    for family in FAMILIES:
        path = (
            FLOW_DIRECTORY
            / f"Malware__{family}__{family}.parquet"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing flow file: {path}"
            )

        frame = pd.read_parquet(
            path,
            columns=[
                "class_name",
                "capture",
                "flow_id",
                "start_time",
                "packet_count",
                "sequence_packets",
                "signed_packet_sizes",
                "interarrival_seconds",
                "valid_ge3"
            ]
        )

        frames.append(
            frame
        )

    source = pd.concat(
        frames,
        ignore_index=True
    )

    source = source[
        source[
            "valid_ge3"
        ]
    ].copy()

    if source.duplicated(
        [
            "class_name",
            "flow_id"
        ]
    ).any():
        raise RuntimeError(
            "Duplicate source keys"
        )

    selected = source.merge(
        manifest[
            [
                "class_name",
                "capture",
                "flow_id",
                "start_time",
                "packet_count",
                "sequence_packets",
                "representation_fingerprint",
                "closed_split",
                "adaptation_partition"
            ]
        ],
        on=[
            "class_name",
            "flow_id"
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_source",
            "_manifest"
        )
    )

    if len(selected) != 229581:
        raise RuntimeError(
            f"Joined rows={len(selected)}"
        )

    if not (
        selected[
            "capture_source"
        ]
        == selected[
            "capture_manifest"
        ]
    ).all():
        raise RuntimeError(
            "Capture mismatch"
        )

    if not (
        selected[
            "packet_count_source"
        ]
        == selected[
            "packet_count_manifest"
        ]
    ).all():
        raise RuntimeError(
            "Packet-count mismatch"
        )

    if not (
        selected[
            "sequence_packets_source"
        ]
        == selected[
            "sequence_packets_manifest"
        ]
    ).all():
        raise RuntimeError(
            "Sequence-length mismatch"
        )

    if not (
        selected[
            "start_time_source"
        ]
        == selected[
            "start_time_manifest"
        ]
    ).all():
        raise RuntimeError(
            "Start-time mismatch"
        )

    size_length_failures = 0
    iat_length_failures = 0

    for (
        expected,
        sizes,
        iats
    ) in zip(
        selected[
            "sequence_packets_source"
        ],
        selected[
            "signed_packet_sizes"
        ],
        selected[
            "interarrival_seconds"
        ]
    ):
        expected = int(
            expected
        )

        if len(
            sizes
        ) != expected:
            size_length_failures += 1

        if len(
            iats
        ) != expected:
            iat_length_failures += 1

    if size_length_failures:
        raise RuntimeError(
            f"Size length failures={size_length_failures}"
        )

    if iat_length_failures:
        raise RuntimeError(
            f"IAT length failures={iat_length_failures}"
        )

    output = pd.DataFrame(
        {
            "class_name": selected[
                "class_name"
            ],
            "capture": selected[
                "capture_manifest"
            ],
            "flow_id": selected[
                "flow_id"
            ],
            "start_time": selected[
                "start_time_manifest"
            ],
            "packet_count": selected[
                "packet_count_manifest"
            ],
            "sequence_packets": selected[
                "sequence_packets_manifest"
            ],
            "signed_packet_sizes": selected[
                "signed_packet_sizes"
            ],
            "interarrival_seconds": selected[
                "interarrival_seconds"
            ],
            "representation_fingerprint": selected[
                "representation_fingerprint"
            ],
            "closed_split": selected[
                "closed_split"
            ],
            "adaptation_partition": selected[
                "adaptation_partition"
            ]
        }
    )

    output = (
        output
        .sort_values(
            [
                "class_name",
                "start_time",
                "flow_id"
            ],
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd"
    )

    print("===== USTC MODEL READY =====")
    print(f"ROWS={len(output)}")
    print(
        f"FAMILIES="
        f"{output['class_name'].nunique()}"
    )

    print()
    print("===== SPLITS =====")

    print(
        output[
            "closed_split"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("===== FAMILY × SPLIT =====")

    print(
        output.groupby(
            [
                "class_name",
                "closed_split"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .to_string()
    )

    print()
    print(
        f"OUTPUT={OUTPUT_PATH}"
    )

    print(
        "USTC_CLOSED_MODEL_READY=PASS"
    )


if __name__ == "__main__":
    main()
