import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import ecu_malnett_apt_validation as original
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecu_malnett_apt_validation import (
    FAMILIES,
    MAX_PACKETS,
    SEED,
    bidirectional_key,
    capture_splits,
    checksum,
    discover_pairs,
    extract_capture,
    inspect_inputs,
    key_from_values,
)

SPLIT_SEEDS = (20261011, 20261012, 20261013, 20261014, 20261015)
OUTPUT_PARENT = Path("artifacts/modern_validation/ecu_malnett_v2")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, rows, columns):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def input_snapshot(pairs, root):
    entries = []
    for _, capture, report in pairs:
        for path in (capture, report):
            details = path.stat()
            entries.append({
                "path": str(path.relative_to(root)),
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
            })
    return sorted(entries, key=lambda row: row["path"])


def reconstruct_frozen_manifest(records, previous_manifest):
    prepared, audit = capture_splits([dict(row) for row in records], SEED)
    actual = {
        (row["family"], row["capture"], row["flow_id"]): row
        for row in prepared
    }
    expected = {
        (row["family"], row["capture"], row["flow_id"]): row
        for row in read_csv(previous_manifest)
    }
    if len(actual) != len(prepared) or len(expected) != len(read_csv(previous_manifest)):
        raise RuntimeError("Duplicate flow identifiers found in one of the manifests")
    if actual.keys() != expected.keys():
        missing = len(expected.keys() - actual.keys())
        added = len(actual.keys() - expected.keys())
        raise RuntimeError(f"Frozen manifest mismatch: missing={missing}, unexpected={added}")
    for key in actual:
        current, earlier = actual[key], expected[key]
        for column in ("split", "fingerprint", "prefix_fingerprint"):
            if current[column] != earlier[column]:
                raise RuntimeError(f"Frozen manifest changed: {key}, field={column}")
        if current["packet_count"] != int(earlier["packet_count"]):
            raise RuntimeError(f"Packet count changed for {key}")
    if audit["removed_records_total"]:
        raise RuntimeError("Unexpected duplicate removal in reconstructed frozen split")
    return prepared


def label_evidence_for_records(pairs, root, records):
    record_ids = {(row["capture"], row["flow_id"]) for row in records}
    evidence = defaultdict(list)
    for _, capture, report in pairs:
        rel = str(capture.relative_to(root))
        with report.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                if str(row.get("malicious", "")).strip() != "1":
                    continue
                direction = key_from_values(
                    row.get("src", ""), row.get("dst", ""),
                    row.get("sport", ""), row.get("dport", ""),
                    row.get("transport", ""),
                )
                if direction is None:
                    continue
                flow_id = checksum(rel + repr(bidirectional_key(direction)))[:24]
                if (rel, flow_id) not in record_ids:
                    continue
                evidence[(rel, flow_id)].append({
                    "line": line,
                    "score": row.get("score", ""),
                    "reasons": row.get("reasons", ""),
                    "packet_count": row.get("packet_count", ""),
                })
    return evidence


def choose_review_samples(records, evidence, per_family):
    result = []
    for family in FAMILIES:
        family_records = [row for row in records if row["family"] == family]
        family_records.sort(key=lambda row: (row["capture"], row["flow_id"]))
        by_capture = defaultdict(list)
        for row in family_records:
            by_capture[row["capture"]].append(row)
        chosen = []
        for capture in sorted(by_capture):
            chosen.append(by_capture[capture][0])
        if len(chosen) > per_family:
            rng = np.random.default_rng(SEED)
            indices = sorted(rng.choice(len(chosen), per_family, replace=False))
            chosen = [chosen[index] for index in indices]
        if len(chosen) < per_family:
            selected = {(row["capture"], row["flow_id"]) for row in chosen}
            remaining = [row for row in family_records if (row["capture"], row["flow_id"]) not in selected]
            chosen.extend(remaining[:per_family - len(chosen)])
        for row in chosen:
            matching = evidence.get((row["capture"], row["flow_id"]), [])
            if not matching:
                raise RuntimeError("No Construct positive row found for retained flow")
            result.append({
                "family": family,
                "capture": row["capture"],
                "flow_id": row["flow_id"],
                "split": row["split"],
                "pcap_bidir_packet_count": row["packet_count"],
                "construct_positive_csv_record_numbers": ";".join(str(item["line"]) for item in matching),
                "construct_directional_packet_counts": ";".join(str(item["packet_count"]) for item in matching),
                "construct_scores": ";".join(str(item["score"]) for item in matching),
                "construct_reasons": " || ".join(str(item["reasons"]) for item in matching)[:1200],
                "review_status": "",
                "review_notes": "",
            })
    return result


def save_sequences(path, records):
    sizes = np.zeros((len(records), MAX_PACKETS), dtype=np.int32)
    iats = np.zeros((len(records), MAX_PACKETS), dtype=np.float64)
    for index, row in enumerate(records):
        length = len(row["sizes"])
        if length < 5 or length > MAX_PACKETS or len(row["iats"]) != length:
            raise RuntimeError(f"Invalid packet sequence: {row['flow_id']}")
        sizes[index, :length] = row["sizes"]
        iats[index, :length] = row["iats"]
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            sizes=sizes,
            iats=iats,
            lengths=np.asarray([len(row["sizes"]) for row in records], dtype=np.int16),
            packet_counts=np.asarray([row["packet_count"] for row in records], dtype=np.int64),
            families=np.asarray([row["family"] for row in records]),
            captures=np.asarray([row["capture"] for row in records]),
            flow_ids=np.asarray([row["flow_id"] for row in records]),
            fingerprints=np.asarray([row["fingerprint"] for row in records]),
            prefix_fingerprints=np.asarray([row["prefix_fingerprint"] for row in records]),
        )


def load_records(path):
    records = []
    with np.load(path, allow_pickle=False) as data:
        for index in range(len(data["families"])):
            length = int(data["lengths"][index])
            records.append({
                "family": str(data["families"][index]),
                "capture": str(data["captures"][index]),
                "flow_id": str(data["flow_ids"][index]),
                "fingerprint": str(data["fingerprints"][index]),
                "prefix_fingerprint": str(data["prefix_fingerprints"][index]),
                "packet_count": int(data["packet_counts"][index]),
                "sizes": data["sizes"][index, :length].tolist(),
                "iats": data["iats"][index, :length].tolist(),
            })
    return records


def audit(args, root, run_dir, previous_run):
    if run_dir.exists():
        raise RuntimeError(f"Refusing to modify existing run directory: {run_dir}")
    if not previous_run.is_dir():
        raise RuntimeError(f"Missing frozen experiment directory: {previous_run}")
    manifest_path = previous_run / "split_manifest.csv"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing frozen manifest: {manifest_path}")
    pairs = discover_pairs(root)
    inventory = inspect_inputs(root, pairs)
    records = []
    matching = []
    for family, capture, report in pairs:
        captured, counters = extract_capture(family, capture, report, root)
        records.extend(captured)
        matching.append(counters)
        print(f"AUDITED {family}/{capture.name}: {len(captured)} eligible positive flows", flush=True)
    records = reconstruct_frozen_manifest(records, manifest_path)
    records.sort(key=lambda row: (row["family"], row["capture"], row["flow_id"]))
    evidence = label_evidence_for_records(pairs, root, records)
    samples = choose_review_samples(records, evidence, args.samples_per_group)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_sequences(run_dir / "cached_flow_sequences.npz", records)
    write_csv(run_dir / "label_review_samples.csv", samples, list(samples[0]))
    write_csv(run_dir / "label_matching_recheck.csv", matching, sorted({key for row in matching for key in row}))
    write_csv(run_dir / "input_inventory.csv", inventory, list(inventory[0]))
    write_json(run_dir / "input_snapshot.json", input_snapshot(pairs, root))
    write_json(run_dir / "audit_summary.json", {
        "dataset": "ECU-MALNETT V2 APT subset",
        "label_provenance": "Construct-inferred, not independently verified malware ground truth",
        "source_groups": list(FAMILIES),
        "capture_count": len(pairs),
        "retained_positive_flows": len(records),
        "flow_counts": dict(Counter(row["family"] for row in records)),
        "review_sample_count": len(samples),
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "previous_script_sha256": sha256_file(Path(original.__file__).resolve()),
        "cache_sha256": sha256_file(run_dir / "cached_flow_sequences.npz"),
        "matched_frozen_manifest": True,
        "review_state": "Pending manual review of CSV-to-PCAP alignment and heuristic-label provenance",
    })
    print(f"AUDIT_PASSED=TRUE FLOWS={len(records)} REVIEW_SAMPLES={len(samples)}")
    print(f"OUTPUT={run_dir}")


def sequence_features(record, horizon):
    sizes = np.asarray(record["sizes"][:horizon], dtype=np.float64)
    iats = np.asarray(record["iats"][:horizon], dtype=np.float64)
    if not 5 <= len(sizes) <= MAX_PACKETS:
        raise RuntimeError("Sequence is outside eligible packet limits")
    if np.any(~np.isfinite(iats)) or np.any(iats < 0):
        raise RuntimeError("Invalid interarrival time")
    signed = np.sign(sizes) * np.log1p(np.abs(sizes))
    timing = np.log1p(iats)
    padded_signed = np.zeros(MAX_PACKETS, dtype=np.float64)
    padded_timing = np.zeros(MAX_PACKETS, dtype=np.float64)
    padded_signed[:len(signed)] = signed
    padded_timing[:len(timing)] = timing
    abs_sizes = np.abs(sizes)
    extras = np.asarray([
        len(sizes) / MAX_PACKETS,
        signed.mean(),
        signed.std(),
        np.median(signed),
        abs_sizes.mean(),
        abs_sizes.std(),
        abs_sizes.max(),
        abs_sizes.sum(),
        np.mean(sizes > 0),
        np.mean(np.sign(sizes[1:]) != np.sign(sizes[:-1])),
        timing.mean(),
        timing.std(),
        np.median(timing),
        np.quantile(timing, 0.9),
        timing.max(),
        timing.sum(),
    ], dtype=np.float64)
    return np.concatenate((padded_signed, padded_timing, extras))


def build_features(records):
    return {
        horizon: np.vstack([sequence_features(row, horizon) for row in records])
        for horizon in (50, 5)
    }


def build_model(name, seed):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
    if name == "LogisticRegression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2500, random_state=seed),
        )
    raise ValueError(name)


def evaluate(model, features, targets, test_idx):
    predictions = model.predict(features[test_idx])
    actual = targets[test_idx]
    return {
        "macro_f1": float(f1_score(actual, predictions, labels=list(range(len(FAMILIES))), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
        "predictions": predictions,
        "confusion": confusion_matrix(actual, predictions, labels=list(range(len(FAMILIES)))).tolist(),
    }


def safe_existing_run(run_dir, previous_run):
    if not run_dir.is_dir():
        raise RuntimeError(f"Run audit first: {run_dir}")
    if run_dir.resolve() == previous_run.resolve():
        raise RuntimeError("Original frozen results must remain read-only")
    for filename in ("baseline_results.csv", "baseline_predictions.csv", "robustness_summary.json"):
        if (run_dir / filename).exists():
            raise RuntimeError(f"Refusing to overwrite: {run_dir / filename}")


def verify_audit_artifacts(run_dir, root, previous_run):
    audit_data = json.loads((run_dir / "audit_summary.json").read_text(encoding="utf-8"))
    previous_script = Path(original.__file__).resolve()
    if audit_data["previous_script_sha256"] != sha256_file(previous_script):
        raise RuntimeError("Original extraction script has changed since the audit")
    if audit_data["frozen_manifest_sha256"] != sha256_file(previous_run / "split_manifest.csv"):
        raise RuntimeError("Frozen manifest has changed since the audit")
    if audit_data["cache_sha256"] != sha256_file(run_dir / "cached_flow_sequences.npz"):
        raise RuntimeError("Cached sequences have changed since the audit")
    pairs = discover_pairs(root)
    if input_snapshot(pairs, root) != json.loads((run_dir / "input_snapshot.json").read_text(encoding="utf-8")):
        raise RuntimeError("Raw dataset input timestamps or sizes have changed since the audit")
    return audit_data


def verify_manual_review(run_dir):
    rows = read_csv(run_dir / "label_review_samples.csv")
    if not rows:
        raise RuntimeError("No human review samples were generated")
    statuses = Counter(str(row["review_status"]).strip().lower() for row in rows)
    if any(str(row["review_status"]).strip().lower() != "checked" for row in rows):
        raise RuntimeError(
            "Manual CSV-to-PCAP review is pending. Inspect each sampled row, "
            "set review_status=checked only after verification, and record problems in review_notes. "
            f"Current statuses: {dict(statuses)}"
        )
    return len(rows)


def run_baselines(run_dir, root, previous_run, seeds):
    safe_existing_run(run_dir, previous_run)
    audit_data = verify_audit_artifacts(run_dir, root, previous_run)
    checked = verify_manual_review(run_dir)
    records = load_records(run_dir / "cached_flow_sequences.npz")
    if len(records) != audit_data["retained_positive_flows"]:
        raise RuntimeError("Audit sample count changed")
    lookup = {(row["family"], row["capture"], row["flow_id"]): i for i, row in enumerate(records)}
    if len(lookup) != len(records):
        raise RuntimeError("Duplicate cached identifiers")
    features = build_features(records)
    targets = np.asarray([FAMILIES.index(row["family"]) for row in records], dtype=np.int64)
    results = []
    predictions = []
    split_rows = []
    family_rows = []
    for split_seed in seeds:
        selected, info = capture_splits([dict(row) for row in records], split_seed)
        selected_ids = {lookup[(row["family"], row["capture"], row["flow_id"])]: row["split"] for row in selected}
        train_idx = np.asarray([index for index, split in selected_ids.items() if split == "train"], dtype=np.int64)
        test_idx = np.asarray([index for index, split in selected_ids.items() if split == "test"], dtype=np.int64)
        val_idx = np.asarray([index for index, split in selected_ids.items() if split == "val"], dtype=np.int64)
        if len({records[i]["capture"] for i in train_idx} & {records[i]["capture"] for i in test_idx}):
            raise RuntimeError("Capture leakage detected")
        for idx, split in selected_ids.items():
            split_rows.append({
                "split_seed": split_seed,
                "family": records[idx]["family"],
                "capture": records[idx]["capture"],
                "flow_id": records[idx]["flow_id"],
                "split": split,
            })
        for model_name in ("RandomForest", "LogisticRegression"):
            for train_horizon in (50, 5):
                model = build_model(model_name, split_seed)
                model.fit(features[train_horizon][train_idx], targets[train_idx])
                for test_horizon in ((50, 5) if train_horizon == 50 else (5,)):
                    evaluation = evaluate(model, features[test_horizon], targets, test_idx)
                    method = f"{model_name}_train{train_horizon}_test{test_horizon}"
                    results.append({
                        "split_seed": split_seed,
                        "model": model_name,
                        "train_horizon": train_horizon,
                        "test_horizon": test_horizon,
                        "evaluation": method,
                        "train_flows": len(train_idx),
                        "validation_flows_unused": len(val_idx),
                        "test_flows": len(test_idx),
                        "macro_f1": evaluation["macro_f1"],
                        "balanced_accuracy": evaluation["balanced_accuracy"],
                    })
                    for idx, prediction in zip(test_idx, evaluation["predictions"]):
                        predictions.append({
                            "split_seed": split_seed,
                            "evaluation": method,
                            "capture": records[idx]["capture"],
                            "flow_id": records[idx]["flow_id"],
                            "actual_group": records[idx]["family"],
                            "predicted_group": FAMILIES[int(prediction)],
                        })
                    for family_index, family in enumerate(FAMILIES):
                        table = evaluation["confusion"]
                        size = sum(table[family_index])
                        family_rows.append({
                            "split_seed": split_seed,
                            "evaluation": method,
                            "family": family,
                            "test_flows": size,
                            "recall": float(table[family_index][family_index] / size) if size else 0.0,
                        })
        print(f"SPLIT_SEED={split_seed} TRAIN={len(train_idx)} VAL={len(val_idx)} TEST={len(test_idx)} COMPLETE", flush=True)
    grouped = defaultdict(list)
    for row in results:
        grouped[row["evaluation"]].append(row["macro_f1"])
    summary = {
        "dataset": "ECU-MALNETT V2 APT subset",
        "experiment": "Capture-disjoint, payload-free classical baselines and split sensitivity",
        "human_rows_checked_for_alignment": checked,
        "labels_are_independently_verified": False,
        "flow_count": len(records),
        "split_seeds": list(seeds),
        "model_selection": "Fixed RF and logistic hyperparameters; held-out validation partition not used for tuning",
        "statistics_unit": "Capture-split repetition, not independent dataset replication",
        "results": {
            key: {
                "mean_macro_f1": float(np.mean(scores)),
                "sample_sd_macro_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                "n_splits": len(scores),
            }
            for key, scores in sorted(grouped.items())
        },
    }
    write_csv(run_dir / "baseline_results.csv", results, list(results[0]))
    write_csv(run_dir / "baseline_predictions.csv", predictions, list(predictions[0]))
    write_csv(run_dir / "baseline_per_group.csv", family_rows, list(family_rows[0]))
    write_csv(run_dir / "repeated_capture_splits.csv", split_rows, list(split_rows[0]))
    write_json(run_dir / "robustness_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"OUTPUT={run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "baselines"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-name", default="apt_v2_followup_20261107")
    parser.add_argument("--previous-run-name", default="apt_v2_full_20261106")
    parser.add_argument("--samples-per-group", type=int, default=5)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    args = parser.parse_args()
    if not args.run_name.startswith("apt_v2_followup_") or "/" in args.run_name or ".." in args.run_name:
        raise SystemExit("Use a new run name starting with apt_v2_followup_")
    if "/" in args.previous_run_name or ".." in args.previous_run_name:
        raise SystemExit("Invalid previous run name")
    if args.samples_per_group < 1:
        raise SystemExit("samples-per-group must be positive")
    if len(set(args.split_seeds)) != len(args.split_seeds):
        raise SystemExit("Split seeds must be unique")
    root_project = Path(__file__).resolve().parents[2]
    dataset_root = args.dataset_root.resolve(strict=True)
    expected_dataset = (root_project / "data" / "raw" / "APT").resolve(strict=True)
    if dataset_root != expected_dataset:
        raise SystemExit(f"Dataset root must be the existing read-only path {expected_dataset}")
    output_parent = root_project / OUTPUT_PARENT
    run_dir = output_parent / args.run_name
    previous_run = output_parent / args.previous_run_name
    if run_dir.resolve() == previous_run.resolve():
        raise SystemExit("Refusing to use the frozen experiment directory for output")
    if args.mode == "audit":
        audit(args, dataset_root, run_dir, previous_run)
    else:
        run_baselines(run_dir, dataset_root, previous_run, args.split_seeds)


if __name__ == "__main__":
    main()
