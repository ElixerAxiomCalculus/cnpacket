import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import py7zr


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ustc-tfc2016"
OUTPUT = ROOT / "data" / "interim" / "ustc-tfc2016" / "pcap"
TEMP = ROOT / "data" / "interim" / "ustc-tfc2016" / "extract_temp"
ARTIFACTS = ROOT / "artifacts" / "phase2"

CAPTURE_SUFFIXES = {
    ".pcap",
    ".pcapng"
}


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def copy_capture(
    source,
    destination_directory,
    class_name,
    traffic_type,
    source_artifact,
    source_kind,
    rows
):
    destination_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        destination_directory
        / source.name
    )

    if destination.exists():
        destination = (
            destination_directory
            / f"{class_name}__{source.name}"
        )

    shutil.copy2(
        source,
        destination
    )

    rows.append(
        {
            "traffic_type": traffic_type,
            "class_name": class_name,
            "source_kind": source_kind,
            "source_artifact": str(
                source_artifact.relative_to(ROOT)
            ),
            "capture": str(
                destination.relative_to(ROOT)
            ),
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "size_mib": round(
                destination.stat().st_size
                / 1024
                / 1024,
                3
            ),
            "sha256": sha256(destination)
        }
    )


def process_direct_capture(
    path,
    traffic_type,
    rows
):
    class_name = path.stem

    destination_directory = (
        OUTPUT
        / traffic_type
        / class_name
    )

    copy_capture(
        path,
        destination_directory,
        class_name,
        traffic_type,
        path,
        "direct_pcap",
        rows
    )


def process_archive(
    path,
    traffic_type,
    rows
):
    class_name = path.stem

    temporary_directory = (
        TEMP
        / traffic_type
        / class_name
    )

    if temporary_directory.exists():
        shutil.rmtree(
            temporary_directory
        )

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    with py7zr.SevenZipFile(
        path,
        mode="r"
    ) as archive:
        archive.extractall(
            path=temporary_directory
        )

    captures = sorted(
        item
        for item in temporary_directory.rglob("*")
        if item.is_file()
        and item.suffix.lower()
        in CAPTURE_SUFFIXES
    )

    if not captures:
        raise RuntimeError(
            f"No PCAP/PCAPNG found in {path}"
        )

    destination_directory = (
        OUTPUT
        / traffic_type
        / class_name
    )

    for capture in captures:
        copy_capture(
            capture,
            destination_directory,
            class_name,
            traffic_type,
            path,
            "7z_archive",
            rows
        )

    shutil.rmtree(
        temporary_directory
    )


def main():
    if OUTPUT.exists():
        shutil.rmtree(
            OUTPUT
        )

    if TEMP.exists():
        shutil.rmtree(
            TEMP
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    TEMP.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for traffic_type in [
        "Benign",
        "Malware"
    ]:
        directory = (
            RAW
            / traffic_type
        )

        if not directory.exists():
            raise RuntimeError(
                f"Missing directory: {directory}"
            )

        for path in sorted(
            directory.iterdir()
        ):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            if suffix in CAPTURE_SUFFIXES:
                process_direct_capture(
                    path,
                    traffic_type,
                    rows
                )

            elif suffix == ".7z":
                process_archive(
                    path,
                    traffic_type,
                    rows
                )

    manifest = (
        ARTIFACTS
        / "ustc_capture_manifest.csv"
    )

    with manifest.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        fieldnames = [
            "traffic_type",
            "class_name",
            "source_kind",
            "source_artifact",
            "capture",
            "filename",
            "size_bytes",
            "size_mib",
            "sha256"
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    class_counts = Counter(
        (
            row["traffic_type"],
            row["class_name"]
        )
        for row in rows
    )

    benign_classes = {
        row["class_name"]
        for row in rows
        if row["traffic_type"] == "Benign"
    }

    malware_classes = {
        row["class_name"]
        for row in rows
        if row["traffic_type"] == "Malware"
    }

    total_bytes = sum(
        row["size_bytes"]
        for row in rows
    )

    print(
        f"TOTAL_CAPTURE_FILES={len(rows)}"
    )

    print(
        f"BENIGN_CLASSES={len(benign_classes)}"
    )

    print(
        f"MALWARE_CLASSES={len(malware_classes)}"
    )

    print(
        f"TOTAL_CLASSES="
        f"{len(benign_classes) + len(malware_classes)}"
    )

    print(
        f"TOTAL_EXTRACTED_GIB="
        f"{total_bytes / 1024 / 1024 / 1024:.3f}"
    )

    print()
    print("CLASS_CAPTURE_COUNTS")

    for key in sorted(
        class_counts
    ):
        traffic_type, class_name = key

        print(
            f"{traffic_type:7s} "
            f"{class_name:20s} "
            f"captures={class_counts[key]}"
        )

    print()
    print(
        f"MANIFEST={manifest}"
    )

    print(
        "USTC_EXTRACTION=PASS"
    )


if __name__ == "__main__":
    main()