import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "iscxvpn2016"
OUTPUT = ROOT / "artifacts" / "phase2"


def classify_activity(filename):
    name = filename.lower()

    if "bittorrent" in name:
        return "p2p"

    if (
        "ftps" in name
        or "sftp" in name
        or "scp" in name
        or "skype_file" in name
        or "skype_files" in name
    ):
        return "file_transfer"

    if (
        "video" in name
        or "netflix" in name
        or "spotify" in name
        or "vimeo" in name
        or "youtube" in name
    ):
        return "streaming"

    if (
        "audio" in name
        or "voipbuster" in name
    ):
        return "voip"

    if "chat" in name:
        return "chat"

    if (
        name.startswith("email")
        or name.startswith("vpn_email")
    ):
        return "email"

    return "unmapped"


def classify_domain(archive_name):
    lower = archive_name.lower()

    if lower.startswith("nonvpn"):
        return "nonvpn"

    if lower.startswith("vpn"):
        return "vpn"

    return "unknown"


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for archive_path in sorted(RAW.glob("*.zip")):
        domain = classify_domain(
            archive_path.name
        )

        with zipfile.ZipFile(
            archive_path,
            "r"
        ) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                suffix = Path(
                    member.filename
                ).suffix.lower()

                if suffix not in {
                    ".pcap",
                    ".pcapng"
                }:
                    continue

                rows.append(
                    {
                        "archive": archive_path.name,
                        "capture": member.filename,
                        "domain": domain,
                        "activity": classify_activity(
                            member.filename
                        ),
                        "compressed_bytes": member.compress_size,
                        "uncompressed_bytes": member.file_size,
                        "uncompressed_mib": round(
                            member.file_size
                            / 1024
                            / 1024,
                            3
                        ),
                        "crc32": f"{member.CRC:08x}"
                    }
                )

    manifest_path = (
        OUTPUT
        / "iscxvpn_capture_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "archive",
                "capture",
                "domain",
                "activity",
                "compressed_bytes",
                "uncompressed_bytes",
                "uncompressed_mib",
                "crc32"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = defaultdict(
        lambda: {
            "captures": 0,
            "bytes": 0
        }
    )

    for row in rows:
        key = (
            row["domain"],
            row["activity"]
        )

        summary[key]["captures"] += 1
        summary[key]["bytes"] += (
            row["uncompressed_bytes"]
        )

    summary_rows = []

    for key in sorted(summary):
        domain, activity = key
        values = summary[key]

        summary_rows.append(
            {
                "domain": domain,
                "activity": activity,
                "captures": values["captures"],
                "uncompressed_gib": round(
                    values["bytes"]
                    / 1024
                    / 1024
                    / 1024,
                    3
                )
            }
        )

    summary_path = (
        OUTPUT
        / "iscxvpn_category_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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

    json_path = (
        OUTPUT
        / "iscxvpn_capture_manifest.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2
        )
        + "\n"
    )

    print(
        f"TOTAL_CAPTURES={len(rows)}"
    )

    for row in summary_rows:
        print(
            f"{row['domain']:7s} "
            f"{row['activity']:14s} "
            f"captures={row['captures']:3d} "
            f"size_gib={row['uncompressed_gib']:.3f}"
        )

    unmapped = [
        row
        for row in rows
        if row["activity"] == "unmapped"
    ]

    print(
        f"UNMAPPED_CAPTURES={len(unmapped)}"
    )

    print(
        f"MANIFEST={manifest_path}"
    )

    print(
        f"SUMMARY={summary_path}"
    )


if __name__ == "__main__":
    main()