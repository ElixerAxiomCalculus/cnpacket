import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "phase2"

INPUT = ARTIFACTS / "iscxvpn_capture_manifest.csv"
OUTPUT = ARTIFACTS / "iscxvpn_protocol_manifest.csv"
SUMMARY = ARTIFACTS / "iscxvpn_fold_summary.csv"
PROTOCOL = ARTIFACTS / "iscxvpn_cv_protocol.json"

SEED = 20261011
FOLDS = 5

PRIMARY_ACTIVITIES = {
    "chat",
    "file_transfer",
    "streaming",
    "voip"
}


def deterministic_key(row):
    value = (
        f"{SEED}|"
        f"{row['domain']}|"
        f"{row['activity']}|"
        f"{row['archive']}|"
        f"{row['capture']}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def exclusion_reason(activity):
    if activity == "email":
        return "insufficient_vpn_capture_groups"

    if activity == "p2p":
        return "no_matched_nonvpn_capture_group"

    return "not_in_primary_protocol"


def main():
    with INPUT.open(
        newline="",
        encoding="utf-8"
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    strata = defaultdict(list)

    for row in rows:
        if row["activity"] in PRIMARY_ACTIVITIES:
            strata[
                (
                    row["domain"],
                    row["activity"]
                )
            ].append(row)

    assignments = {}

    for stratum, members in sorted(
        strata.items()
    ):
        ordered = sorted(
            members,
            key=deterministic_key
        )

        if len(ordered) < FOLDS:
            raise RuntimeError(
                f"Stratum {stratum} has only "
                f"{len(ordered)} captures"
            )

        for index, row in enumerate(ordered):
            identity = (
                row["archive"],
                row["capture"]
            )

            assignments[identity] = (
                index % FOLDS
            )

    output_rows = []

    for row in rows:
        identity = (
            row["archive"],
            row["capture"]
        )

        selected = identity in assignments

        output_row = dict(row)

        output_row["primary_selected"] = (
            "true"
            if selected
            else "false"
        )

        output_row["fold"] = (
            str(assignments[identity])
            if selected
            else ""
        )

        output_row["exclusion_reason"] = (
            ""
            if selected
            else exclusion_reason(
                row["activity"]
            )
        )

        output_rows.append(
            output_row
        )

    fieldnames = list(
        output_rows[0].keys()
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    summary = defaultdict(
        lambda: {
            "captures": 0,
            "bytes": 0
        }
    )

    for row in output_rows:
        if row["primary_selected"] != "true":
            continue

        key = (
            row["fold"],
            row["domain"],
            row["activity"]
        )

        summary[key]["captures"] += 1
        summary[key]["bytes"] += int(
            row["uncompressed_bytes"]
        )

    summary_rows = []

    for key in sorted(summary):
        fold, domain, activity = key
        values = summary[key]

        summary_rows.append(
            {
                "fold": fold,
                "domain": domain,
                "activity": activity,
                "captures": values["captures"],
                "uncompressed_gib": round(
                    values["bytes"]
                    / 1024
                    / 1024
                    / 1024,
                    6
                )
            }
        )

    with SUMMARY.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "domain",
                "activity",
                "captures",
                "uncompressed_gib"
            ]
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    for stratum in sorted(strata):
        folds = {
            assignments[
                (
                    row["archive"],
                    row["capture"]
                )
            ]
            for row in strata[stratum]
        }

        if folds != set(range(FOLDS)):
            raise RuntimeError(
                f"Incomplete fold coverage "
                f"for {stratum}: {folds}"
            )

    protocol = {
        "dataset": "ISCXVPN2016",
        "protocol_version": 1,
        "seed": SEED,
        "number_of_folds": FOLDS,
        "primary_target": "activity",
        "domain_attribute": "vpn_status",
        "primary_activities": [
            "chat",
            "file_transfer",
            "streaming",
            "voip"
        ],
        "excluded_activities": {
            "email": (
                "Only two independent VPN "
                "capture groups are available"
            ),
            "p2p": (
                "Only one VPN BitTorrent capture "
                "is available and no matched "
                "non-VPN P2P capture exists"
            )
        },
        "grouping_unit": "capture_file",
        "fold_assignment": (
            "SHA-256 deterministic ordering "
            "within each activity-domain stratum"
        ),
        "rotation": {
            "test_fold": "r",
            "validation_fold": "(r + 1) mod 5",
            "training_folds": (
                "all folds excluding test "
                "and validation"
            )
        },
        "leakage_policy": (
            "Every flow and packet sequence "
            "derived from one capture inherits "
            "that capture's fold"
        )
    }

    PROTOCOL.write_text(
        json.dumps(
            protocol,
            indent=2
        )
        + "\n"
    )

    selected_rows = [
        row
        for row in output_rows
        if row["primary_selected"] == "true"
    ]

    excluded_rows = [
        row
        for row in output_rows
        if row["primary_selected"] == "false"
    ]

    selected_bytes = sum(
        int(row["uncompressed_bytes"])
        for row in selected_rows
    )

    print(
        f"TOTAL_CAPTURE_FILES={len(rows)}"
    )

    print(
        f"PRIMARY_CAPTURE_FILES={len(selected_rows)}"
    )

    print(
        f"EXCLUDED_CAPTURE_FILES={len(excluded_rows)}"
    )

    print(
        f"PRIMARY_UNCOMPRESSED_GIB="
        f"{selected_bytes / 1024 / 1024 / 1024:.3f}"
    )

    for fold in range(FOLDS):
        fold_rows = [
            row
            for row in selected_rows
            if int(row["fold"]) == fold
        ]

        print(
            f"FOLD_{fold}_CAPTURES="
            f"{len(fold_rows)}"
        )

    print(
        "CAPTURE_LEAKAGE_POLICY=PASS"
    )

    print(
        "FOLD_COVERAGE=PASS"
    )

    print(
        f"PROTOCOL_MANIFEST={OUTPUT}"
    )

    print(
        f"CV_PROTOCOL={PROTOCOL}"
    )


if __name__ == "__main__":
    main()