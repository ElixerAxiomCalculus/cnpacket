import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUTPUT_JSON = ROOT / "artifacts" / "phase2" / "raw_dataset_inventory.json"
OUTPUT_CSV = ROOT / "artifacts" / "phase2" / "raw_dataset_inventory.csv"


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def inventory_dataset(dataset_directory):
    rows = []

    for path in sorted(dataset_directory.rglob("*")):
        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue
        
        if "quarantine" in path.parts:
            continue

        relative = path.relative_to(ROOT)

        rows.append(
            {
                "dataset": dataset_directory.name,
                "path": str(relative),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "size_mib": round(
                    path.stat().st_size / 1024 / 1024,
                    3
                ),
                "sha256": sha256(path)
            }
        )

    return rows


def main():
    datasets = [
        RAW / "cicids2017",
        RAW / "ustc-tfc2016",
        RAW / "iscxvpn2016"
    ]

    all_rows = []

    for dataset in datasets:
        if not dataset.exists():
            continue

        all_rows.extend(
            inventory_dataset(dataset)
        )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            all_rows,
            indent=2
        )
        + "\n"
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        fieldnames = [
            "dataset",
            "path",
            "filename",
            "extension",
            "size_bytes",
            "size_mib",
            "sha256"
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_rows)

    totals = {}

    for row in all_rows:
        dataset = row["dataset"]

        totals.setdefault(
            dataset,
            {
                "files": 0,
                "bytes": 0
            }
        )

        totals[dataset]["files"] += 1
        totals[dataset]["bytes"] += row["size_bytes"]

    print("RAW_DATASET_AUDIT")

    for dataset, values in sorted(totals.items()):
        gib = values["bytes"] / 1024 / 1024 / 1024

        print(
            f"{dataset}: "
            f"files={values['files']} "
            f"size_gib={gib:.3f}"
        )

    print(
        f"INVENTORY_JSON={OUTPUT_JSON}"
    )

    print(
        f"INVENTORY_CSV={OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()