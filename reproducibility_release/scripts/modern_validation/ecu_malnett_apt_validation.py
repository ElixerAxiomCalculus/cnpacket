import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import wilcoxon
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

FAMILIES = ("APT28", "APT29", "Lazarus", "RustyStealer", "Turla")
MAX_PACKETS = 50
MIN_PACKETS = 5
SEED = 20261011


def checksum(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key_from_values(src, dst, sport, dport, transport):
    try:
        port_a, port_b = str(int(sport)), str(int(dport))
    except (ValueError, TypeError):
        return None
    proto = transport.strip().upper()
    if proto not in ("TCP", "UDP") or not src or not dst:
        return None
    return (src.strip(), dst.strip(), port_a, port_b, proto)


def reverse_key(key):
    return (key[1], key[0], key[3], key[2], key[4])


def bidirectional_key(key):
    other = reverse_key(key)
    return min(key, other)


def csv_inventory(path):
    count = Counter()
    columns = ()
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        for row in reader:
            count[str(row.get("malicious", "missing")).strip()] += 1
    return columns, count


def discover_pairs(root):
    pairs = []
    for family in FAMILIES:
        capture_dir = root / "pcaps" / family
        report_dir = root / f"analysis_output_{family.lower()}_baseline_labelled"
        if not capture_dir.is_dir() or not report_dir.is_dir():
            raise RuntimeError(f"Missing required dataset directory: {capture_dir} or {report_dir}")
        captures = sorted(capture_dir.rglob("*.pcap")) + sorted(capture_dir.rglob("*.pcapng"))
        if not captures:
            raise RuntimeError(f"No PCAP files found for {family}")
        for capture in captures:
            report = report_dir / f"{capture.stem}_sessions.csv"
            if not report.is_file():
                raise RuntimeError(f"Missing Construct label report for {capture}: {report}")
            pairs.append((family, capture, report))
    return pairs


def inspect_inputs(root, pairs):
    summary = []
    required = {"src", "dst", "sport", "dport", "transport", "packet_count", "malicious"}
    for family, capture, report in pairs:
        columns, counts = csv_inventory(report)
        absent = required - set(columns)
        if absent:
            raise RuntimeError(f"Missing Construct columns {sorted(absent)} in {report}")
        summary.append({"family": family, "capture": str(capture.relative_to(root)), "pcap_bytes": capture.stat().st_size,
                        "label_file": str(report.relative_to(root)), "positive_csv_rows": counts["1"],
                        "negative_csv_rows": counts["0"], "other_label_rows": sum(v for k, v in counts.items() if k not in ("0", "1"))})
    print(json.dumps({"dataset": str(root), "capture_count": len(pairs),
                      "families": dict(Counter(p["family"] for p in summary)),
                      "total_pcap_bytes": sum(p["pcap_bytes"] for p in summary),
                      "positive_csv_rows": sum(p["positive_csv_rows"] for p in summary),
                      "negative_csv_rows": sum(p["negative_csv_rows"] for p in summary)}, indent=2))
    return summary


def read_construct_labels(path):
    mapping = {}
    ambiguous = set()
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        for row in csv.DictReader(handle):
            key = key_from_values(row.get("src", ""), row.get("dst", ""), row.get("sport", ""), row.get("dport", ""), row.get("transport", ""))
            label = str(row.get("malicious", "")).strip()
            if key is None or label not in ("0", "1"):
                continue
            try:
                packet_count = int(float(row["packet_count"]))
            except (TypeError, ValueError):
                continue
            value = (int(label), packet_count)
            if key in mapping:
                ambiguous.add(key)
            else:
                mapping[key] = value
    for key in ambiguous:
        del mapping[key]
    return mapping, len(ambiguous)


def tshark_packets(path):
    if shutil.which("tshark") is None:
        raise RuntimeError("TShark not found. Install it in WSL before running the experiment.")
    fields = ("frame.time_epoch", "frame.len", "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
              "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport")
    command = ["tshark", "-n", "-r", str(path), "-Y", "tcp || udp", "-T", "fields", "-E", "occurrence=f"]
    for field in fields:
        command.extend(("-e", field))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            row = line.rstrip("\n").split("\t")
            if len(row) != len(fields):
                continue
            stamp, size, ip_src, ip_dst, ip6_src, ip6_dst, tcp_src, tcp_dst, udp_src, udp_dst = row
            source = ip_src or ip6_src
            destination = ip_dst or ip6_dst
            if tcp_src and tcp_dst:
                transport, source_port, destination_port = "TCP", tcp_src, tcp_dst
            elif udp_src and udp_dst:
                transport, source_port, destination_port = "UDP", udp_src, udp_dst
            else:
                continue
            key = key_from_values(source, destination, source_port, destination_port, transport)
            if key is None:
                continue
            try:
                yield float(stamp), int(size), key
            except ValueError:
                continue
    finally:
        if process.stdout:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr else ""
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"TShark failed on {path}: {stderr[-1500:]}")


def extract_capture(family, capture, report, root):
    labels, ambiguous = read_construct_labels(report)
    flows = {}
    observed = Counter()
    ignored = Counter()
    eligible_bidir_keys = {bidirectional_key(direction) for direction in labels}
    for timestamp, frame_length, direction in tshark_packets(capture):
        key = bidirectional_key(direction)
        if key not in eligible_bidir_keys:
            ignored["unlabelled_packet"] += 1
            continue
        observed[direction] += 1
        state = flows.get(key)
        if state is None:
            state = {"first_direction": direction, "last_time": timestamp, "total": 0, "long_gap": False,
                     "out_of_order": False, "sizes": [], "iats": [], "directions": set()}
            flows[key] = state
        delta = timestamp - state["last_time"]
        if state["total"] and delta > 120.0:
            state["long_gap"] = True
        if delta < -0.000001:
            state["out_of_order"] = True
        state["last_time"] = timestamp
        state["total"] += 1
        state["directions"].add(direction)
        if len(state["sizes"]) < MAX_PACKETS:
            sign = 1 if direction == state["first_direction"] else -1
            state["sizes"].append(sign * frame_length)
            state["iats"].append(0.0 if state["total"] == 1 else max(0.0, delta))
    for key, n in observed.items():
        if labels[key][1] != n:
            ignored["directional_packet_count_mismatch"] += 1
    records = []
    for key, state in flows.items():
        directions = state["directions"]
        if any(direction not in labels or labels[direction][1] != observed[direction] for direction in directions):
            ignored["mismatched_flow"] += 1
            continue
        if any(labels[direction][0] != 1 for direction in directions):
            ignored["not_all_directions_positive"] += 1
            continue
        if state["long_gap"] or state["out_of_order"]:
            ignored["ambiguous_timing"] += 1
            continue
        if state["total"] < MIN_PACKETS:
            ignored["short_flow"] += 1
            continue
        if state["total"] > 0 and not state["sizes"]:
            continue
        capture_id = str(capture.relative_to(root))
        fingerprint = checksum(json.dumps([len(state["sizes"]), state["sizes"],
                                            [round(v * 1000000) for v in state["iats"]]], separators=(",", ":")))
        prefix_fingerprint = checksum(json.dumps([state["sizes"][:5],
                                                   [round(v * 1000000) for v in state["iats"][:5]]], separators=(",", ":")))
        records.append({"family": family, "capture": capture_id, "flow_id": checksum(capture_id + repr(key))[:24],
                        "fingerprint": fingerprint, "prefix_fingerprint": prefix_fingerprint, "packet_count": state["total"],
                        "sizes": state["sizes"], "iats": state["iats"]})
    audit = {"family": family, "capture": str(capture.relative_to(root)), "labelled_directional_keys": len(labels),
             "duplicate_report_keys": ambiguous, "bidirectional_flows_seen": len(flows), "retained_positive_flows": len(records),
             **dict(ignored)}
    return records, audit


def capture_splits(records, seed):
    result = {}
    rng = np.random.default_rng(seed)
    for family in FAMILIES:
        captures = sorted({r["capture"] for r in records if r["family"] == family})
        if len(captures) < 3:
            raise RuntimeError(f"{family} has only {len(captures)} captures with eligible labelled flows; minimum is three")
        captures = list(rng.permutation(captures))
        test_count = max(1, round(len(captures) * 0.2))
        val_count = max(1, round(len(captures) * 0.2))
        if test_count + val_count >= len(captures):
            raise RuntimeError(f"Not enough independent captures for {family}")
        for i, capture in enumerate(captures):
            result[capture] = "test" if i < test_count else "val" if i < test_count + val_count else "train"
    conflict = defaultdict(lambda: {"families": set(), "splits": set()})
    for record in records:
        split = result[record["capture"]]
        for name in ("fingerprint", "prefix_fingerprint"):
            item = conflict[(name, record[name])]
            item["families"].add(record["family"])
            item["splits"].add(split)
    invalid = {key for key, group in conflict.items() if len(group["families"]) != 1 or len(group["splits"]) != 1}
    keep = []
    seen = set()
    for record in records:
        keys = (("fingerprint", record["fingerprint"]), ("prefix_fingerprint", record["prefix_fingerprint"]))
        if all(key not in invalid and key not in seen for key in keys):
            record["split"] = result[record["capture"]]
            keep.append(record)
            seen.update(keys)
    for family in FAMILIES:
        for split in ("train", "val", "test"):
            if not any(r["family"] == family and r["split"] == split for r in keep):
                raise RuntimeError(f"No eligible {family}/{split} flows remain after deduplication")
    return keep, {"removed_conflicting_full_or_prefix_fingerprints": len(invalid),
                  "removed_records_total": len(records) - len(keep), "capture_assignment": result}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalization(records):
    ss, si, sii, count = 0.0, 0.0, 0.0, 0
    for record in records:
        a = np.asarray(record["sizes"], dtype=np.float64)
        b = np.log1p(np.maximum(0.0, np.asarray(record["iats"], dtype=np.float64)))
        ss += float(a @ a)
        si += float(b.sum())
        sii += float(b @ b)
        count += len(a)
    if not count:
        raise RuntimeError("Cannot normalize empty training set")
    size_scale = max(math.sqrt(ss / count), 1e-8)
    mean = si / count
    std = max(math.sqrt(max(0.0, sii / count - mean * mean)), 1e-8)
    return size_scale, mean, std


def encode(records, norm, class_map, horizon=50):
    size_scale, iat_mean, iat_std = norm
    data = np.zeros((len(records), MAX_PACKETS, 2), dtype=np.float32)
    lengths = np.empty(len(records), dtype=np.int64)
    labels = np.empty(len(records), dtype=np.int64)
    for i, record in enumerate(records):
        length = min(horizon, len(record["sizes"]))
        lengths[i] = length
        data[i, :length, 0] = np.asarray(record["sizes"][:length], dtype=np.float32) / size_scale
        data[i, :length, 1] = (np.log1p(np.maximum(0.0, np.asarray(record["iats"][:length], dtype=np.float32))) - iat_mean) / iat_std
        labels[i] = class_map.get(record["family"], -1)
    return data, lengths, labels


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        f = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        mask = (torch.arange(f.shape[1], device=f.device)[None, :] < lengths.to(f.device)[:, None]).unsqueeze(-1)
        f = f * mask
        packed = pack_padded_sequence(f, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)


class Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = Encoder()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, classes)

    def forward(self, x, lengths):
        return self.classifier(self.dropout(self.encoder(x, lengths)))


def dataloader(x, lengths, labels, batch_size, shuffle=False, sampler=None):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(lengths), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=0, pin_memory=False)


def predict_logits(model, data, device, batch_size):
    x, lengths, _ = data
    dummy = np.zeros(len(lengths), dtype=np.int64)
    batches = dataloader(x, lengths, dummy, batch_size)
    result = []
    model.eval()
    with torch.no_grad():
        for xb, lens, _ in batches:
            result.append(model(xb.to(device), lens).cpu().numpy())
    return np.concatenate(result) if result else np.zeros((0, model.classifier.out_features))


def embeddings(model, data, device, batch_size):
    x, lengths, _ = data
    dummy = np.zeros(len(lengths), dtype=np.int64)
    batches = dataloader(x, lengths, dummy, batch_size)
    result = []
    model.eval()
    with torch.no_grad():
        for xb, lens, _ in batches:
            z = model(xb.to(device), lens)
            result.append(F.normalize(z, dim=1).cpu().numpy())
    return np.concatenate(result) if result else np.zeros((0, 128), dtype=np.float32)


def train_classifier(train, val, classes, seed, device, epochs, batch_size, patience_limit=10):
    set_seed(seed)
    model = Classifier(classes).to(device)
    counts = np.bincount(train[2], minlength=classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (classes * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    batches = dataloader(*train, batch_size, shuffle=True)
    best, best_score, patience = None, -1.0, 0
    for epoch in range(epochs):
        model.train()
        for xb, lengths, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), lengths)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = np.argmax(predict_logits(model, val, device, batch_size), axis=1)
        score = f1_score(val[2], prediction, average="macro", labels=list(range(classes)), zero_division=0)
        if score > best_score + 1e-4:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break
    model.load_state_dict(best)
    return model, {"best_val_macro_f1": float(best_score), "epochs_run": epoch + 1}


def supcon_loss(z, labels, temperature=0.07):
    z = F.normalize(z, dim=1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    neg = torch.exp(logits).masked_fill(eye, 0.0).sum(dim=1).clamp_min(1e-12)
    per_sample = -(logits - neg.log()[:, None]).masked_fill(~positives, 0.0).sum(dim=1)
    num = positives.sum(dim=1)
    valid = num > 0
    if not valid.any():
        raise RuntimeError("SupCon batch had no positive pairs")
    return (per_sample[valid] / num[valid]).mean()


def prototypes(z, labels, classes):
    out = []
    for c in range(classes):
        subset = z[labels == c]
        if not len(subset):
            raise RuntimeError(f"No prototype training samples for class {c}")
        p = subset.mean(axis=0)
        out.append(p / max(np.linalg.norm(p), 1e-12))
    return np.stack(out)


def train_supcon(train, val, seed, device, epochs, batch_size):
    set_seed(seed)
    model = Encoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    counts = np.bincount(train[2]); weights = 1.0 / np.maximum(counts[train[2]], 1)
    sampler = WeightedRandomSampler(torch.from_numpy(weights.astype(np.float64)), len(weights), replacement=True)
    batches = dataloader(*train, batch_size, sampler=sampler)
    best, best_score, patience = None, -1.0, 0
    classes = len(counts)
    for epoch in range(epochs):
        model.train()
        for xb, lengths, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = supcon_loss(model(xb.to(device), lengths), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ztrain = embeddings(model, train, device, batch_size)
        p = prototypes(ztrain, train[2], classes)
        zval = embeddings(model, val, device, batch_size)
        pred = np.argmax(zval @ p.T, axis=1)
        score = f1_score(val[2], pred, average="macro", labels=list(range(classes)), zero_division=0)
        if score > best_score + 1e-4:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 8:
            break
    model.load_state_dict(best)
    return model, {"best_val_macro_f1": float(best_score), "epochs_run": epoch + 1}


def unknown_scores(mode, model, data, device, batch_size, prototype_matrix=None):
    if mode == "CE+MSP":
        scores = predict_logits(model, data, device, batch_size)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        return 1.0 - p.max(axis=1), p.argmax(axis=1)
    z = embeddings(model, data, device, batch_size)
    sim = z @ prototype_matrix.T
    return 1.0 - sim.max(axis=1), sim.argmax(axis=1)


def calibrate(known, novel):
    truth = np.r_[np.zeros(len(known), dtype=np.int64), np.ones(len(novel), dtype=np.int64)]
    scores = np.r_[known, novel]
    fpr, tpr, thresholds = roc_curve(truth, scores)
    candidates = np.isfinite(thresholds)
    if not candidates.any():
        raise RuntimeError("No finite calibration threshold available")
    objective = np.where(candidates, tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(objective))])


def open_metrics(known_scores, unknown_scores_, known_pred, unknown_pred, known_labels, threshold):
    truth = np.r_[np.zeros(len(known_scores), dtype=int), np.ones(len(unknown_scores_), dtype=int)]
    scores = np.r_[known_scores, unknown_scores_]
    auroc = roc_auc_score(truth, scores)
    unknown_label = max(int(known_labels.max()), int(known_pred.max())) + 1
    predicted = np.r_[np.where(known_scores >= threshold, unknown_label, known_pred),
                      np.where(unknown_scores_ >= threshold, unknown_label, unknown_pred)]
    actual = np.r_[known_labels, np.full(len(unknown_scores_), unknown_label)]
    valid_labels = list(range(unknown_label + 1))
    return {"unknown_auroc": float(auroc),
            "known_false_rejection": float(np.mean(known_scores >= threshold)),
            "unknown_recall": float(np.mean(unknown_scores_ >= threshold)),
            "open_macro_f1": float(f1_score(actual, predicted, average="macro", labels=valid_labels, zero_division=0))}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subset(records, family=None, split=None, exclude=()):
    return [r for r in records if (family is None or r["family"] == family) and
            (split is None or r["split"] == split) and r["family"] not in exclude]


def run_experiments(records, args, out):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device selected but unavailable")
    info = {"device": device, "closed_epochs": args.closed_epochs, "open_epochs": args.open_epochs,
            "ce_batch_size": args.batch_size, "supcon_batch_size": args.supcon_batch_size, "seeds": args.seeds}
    closed_results = []
    class_map = {family: i for i, family in enumerate(FAMILIES)}
    train_records = subset(records, split="train")
    val_records = subset(records, split="val")
    test_records = subset(records, split="test")
    norm = normalization(train_records)
    train = encode(train_records, norm, class_map)
    val = encode(val_records, norm, class_map)
    test50 = encode(test_records, norm, class_map, 50)
    test5 = encode(test_records, norm, class_map, 5)
    for seed in args.seeds:
        model, training = train_classifier(train, val, 5, seed, device, args.closed_epochs, args.batch_size)
        for horizon, data in ((50, test50), (5, test5)):
            pred = np.argmax(predict_logits(model, data, device, args.batch_size), axis=1)
            closed_results.append({"seed": seed, "horizon": horizon,
                                   "macro_f1": float(f1_score(data[2], pred, average="macro", labels=list(range(5)), zero_division=0)),
                                   "balanced_accuracy": float(balanced_accuracy_score(data[2], pred)),
                                   "test_flows": len(data[2]), **training})
        print(f"CLOSED_SET seed={seed}: done", flush=True)
    write_csv(out / "closed_set_results.csv", closed_results)
    del model, train, val, test50, test5
    rotation_results, fewshot_results = [], []
    for rotation, target in enumerate(FAMILIES):
        calibration_family = FAMILIES[(rotation + 1) % len(FAMILIES)]
        known = [f for f in FAMILIES if f not in (target, calibration_family)]
        cm = {family: i for i, family in enumerate(known)}
        known_train = subset(records, split="train", exclude=(target, calibration_family))
        known_val = subset(records, split="val", exclude=(target, calibration_family))
        known_test = subset(records, split="test", exclude=(target, calibration_family))
        cal_val = subset(records, family=calibration_family, split="val")
        target_test = subset(records, family=target, split="test")
        target_support = subset(records, family=target, split="train")
        norm = normalization(known_train)
        t = encode(known_train, norm, cm)
        v = encode(known_val, norm, cm)
        kt50 = encode(known_test, norm, cm, 50)
        kt5 = encode(known_test, norm, cm, 5)
        cv50 = encode(cal_val, norm, cm, 50)
        cv5 = encode(cal_val, norm, cm, 5)
        ut50 = encode(target_test, norm, cm, 50)
        ut5 = encode(target_test, norm, cm, 5)
        seed = args.seeds[0]
        for method in ("CE+MSP", "SupCon+Prototype"):
            if method == "CE+MSP":
                model, training = train_classifier(t, v, 3, seed, device, args.open_epochs, args.batch_size, patience_limit=8)
                proto = None
                predictor = model
            else:
                model, training = train_supcon(t, v, seed, device, args.open_epochs, args.supcon_batch_size)
                proto = prototypes(embeddings(model, t, device, args.batch_size), t[2], 3)
                predictor = model
            v50_sc, _ = unknown_scores("CE+MSP" if method == "CE+MSP" else "SUPCON", predictor, v, device, args.batch_size, proto)
            c50_sc, _ = unknown_scores("CE+MSP" if method == "CE+MSP" else "SUPCON", predictor, cv50, device, args.batch_size, proto)
            v5 = encode(known_val, norm, cm, 5)
            v5_sc, _ = unknown_scores("CE+MSP" if method == "CE+MSP" else "SUPCON", predictor, v5, device, args.batch_size, proto)
            c5_sc, _ = unknown_scores("CE+MSP" if method == "CE+MSP" else "SUPCON", predictor, cv5, device, args.batch_size, proto)
            threshold_original = calibrate(v50_sc, c50_sc)
            threshold_prefix = calibrate(v5_sc, c5_sc)
            for horizon, known_data, novel_data in ((50, kt50, ut50), (5, kt5, ut5)):
                kscore, kpred = unknown_scores("CE+MSP" if method == "CE+MSP" else "SUPCON", predictor, known_data, device, args.batch_size, proto)
                uscore, upred = unknown_scores("CE+MSP" if method == "CE+MSP" else "SUPCON", predictor, novel_data, device, args.batch_size, proto)
                threshold = threshold_original
                vals = open_metrics(kscore, uscore, kpred, upred, known_data[2], threshold)
                rotation_results.append({"rotation": rotation, "target": target, "calibration": calibration_family,
                                         "known_families": ",".join(known), "method": method, "horizon": horizon,
                                         "threshold_policy": "original_50_packet", "threshold": threshold,
                                         "known_test_flows": len(known_data[2]), "unknown_test_flows": len(novel_data[2]),
                                         **vals, **training})
                if horizon == 5:
                    vals = open_metrics(kscore, uscore, kpred, upred, known_data[2], threshold_prefix)
                    rotation_results.append({"rotation": rotation, "target": target, "calibration": calibration_family,
                                             "known_families": ",".join(known), "method": method, "horizon": horizon,
                                             "threshold_policy": "recalibrated_5_packet", "threshold": threshold_prefix,
                                             "known_test_flows": len(known_data[2]), "unknown_test_flows": len(novel_data[2]),
                                             **vals, **training})
            if method == "SupCon+Prototype":
                if len(target_support) >= 10:
                    indices = np.random.default_rng(SEED + rotation).permutation(len(target_support))[:10]
                    support_data = encode([target_support[int(i)] for i in indices], norm, cm)
                    target_z = embeddings(model, support_data, device, args.batch_size)
                    known_z = embeddings(model, kt50, device, args.batch_size)
                    query_z = embeddings(model, ut50, device, args.batch_size)
                    for k in (1, 5, 10):
                        p = target_z[:k].mean(axis=0)
                        p /= max(np.linalg.norm(p), 1e-12)
                        all_protos = np.vstack((proto, p))
                        known_pred = (known_z @ all_protos.T).argmax(axis=1)
                        target_pred = (query_z @ all_protos.T).argmax(axis=1)
                        actual = np.r_[kt50[2], np.full(len(target_pred), 3)]
                        pred = np.r_[known_pred, target_pred]
                        fewshot_results.append({"rotation": rotation, "target": target, "k": k,
                                                "target_f1": float(f1_score(actual == 3, pred == 3, zero_division=0)),
                                                "four_class_macro_f1": float(f1_score(actual, pred, average="macro", labels=list(range(4)), zero_division=0)),
                                                "known_test_flows": len(known_pred), "target_test_flows": len(target_pred)})
                else:
                    print(f"FEWSHOT_SKIP target={target}: fewer than 10 support flows", flush=True)
            print(f"OPEN_SET target={target} method={method}: done", flush=True)
            del model
        write_csv(out / "open_set_results.csv", rotation_results)
        write_csv(out / "fewshot_results.csv", fewshot_results)
    write_csv(out / "closed_set_results.csv", closed_results)
    write_csv(out / "open_set_results.csv", rotation_results)
    write_csv(out / "fewshot_results.csv", fewshot_results)
    summary = {"environment": info, "closed_set": {}, "open_set": {}, "fewshot": {}}
    for horizon in (50, 5):
        values = [r["macro_f1"] for r in closed_results if r["horizon"] == horizon]
        summary["closed_set"][str(horizon)] = {"macro_f1_mean": float(np.mean(values)), "macro_f1_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n": len(values)}
    for method in ("CE+MSP", "SupCon+Prototype"):
        for horizon in (50, 5):
            a = [r for r in rotation_results if r["method"] == method and r["horizon"] == horizon and r["threshold_policy"] == "original_50_packet"]
            summary["open_set"][f"{method}_{horizon}"] = {"mean_unknown_auroc": float(np.mean([r["unknown_auroc"] for r in a])),
                                                               "sample_sd": float(np.std([r["unknown_auroc"] for r in a], ddof=1)), "n": len(a)}
    for horizon in (50, 5):
        ce = {r["target"]: r["unknown_auroc"] for r in rotation_results if r["method"] == "CE+MSP" and
              r["horizon"] == horizon and r["threshold_policy"] == "original_50_packet"}
        sup = {r["target"]: r["unknown_auroc"] for r in rotation_results if r["method"] == "SupCon+Prototype" and
               r["horizon"] == horizon and r["threshold_policy"] == "original_50_packet"}
        paired = np.array([sup[f] - ce[f] for f in FAMILIES], dtype=np.float64)
        if np.allclose(paired, 0):
            w, p = 0.0, 1.0
        else:
            test = wilcoxon(paired, alternative="two-sided", method="auto")
            w, p = float(test.statistic), float(test.pvalue)
        summary.setdefault("paired_comparison", {})[str(horizon)] = {
            "n_target_groups": len(FAMILIES), "mean_auroc_difference": float(paired.mean()),
            "wilcoxon_w": w, "wilcoxon_two_sided_p": p,
            "interpretation": "Exploratory five-group paired comparison; low statistical power"}
    for k in (1, 5, 10):
        a = [r["target_f1"] for r in fewshot_results if r["k"] == k]
        summary["fewshot"][str(k)] = {"mean_target_f1": float(np.mean(a)) if a else None, "n": len(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figures(closed_results, rotation_results, out)
    print("SUMMARY=" + json.dumps(summary, indent=2), flush=True)



def make_figures(closed_results, rotation_results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = [np.mean([r["macro_f1"] for r in closed_results if r["horizon"] == h]) for h in (50, 5)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(["Up to 50 packets", "First 5 packets"], means)
    ax.set_ylabel("Macro-F1 on the same held-out capture flows")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "matched_horizon_macro_f1.png", dpi=300)
    plt.close(fig)
    for horizon in (50, 5):
        ce = [next(r["unknown_auroc"] for r in rotation_results if r["method"] == "CE+MSP" and
                   r["target"] == f and r["horizon"] == horizon and r["threshold_policy"] == "original_50_packet") for f in FAMILIES]
        sup = [next(r["unknown_auroc"] for r in rotation_results if r["method"] == "SupCon+Prototype" and
                    r["target"] == f and r["horizon"] == horizon and r["threshold_policy"] == "original_50_packet") for f in FAMILIES]
        x = np.arange(len(FAMILIES))
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.bar(x - 0.18, ce, width=0.36, label="CE + MSP")
        ax.bar(x + 0.18, sup, width=0.36, label="SupCon + prototypes")
        ax.axhline(0.5, linestyle="--", linewidth=1, label="Chance reference")
        ax.set_xticks(x, FAMILIES, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Unknown-source-group AUROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"open_set_auroc_{horizon}_packet.png", dpi=300)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="ECU-MALNETT V2 APT capture-disjoint validation")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--dataset-root", type=Path, default=project_root / "data/raw/APT")
    parser.add_argument("--mode", choices=("inventory", "run"), required=True)
    parser.add_argument("--output-root", type=Path, default=project_root / "artifacts/modern_validation/ecu_malnett_v2")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--closed-epochs", type=int, default=60)
    parser.add_argument("--open-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--supcon-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261011, 20261012, 20261013])
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    pairs = discover_pairs(root)
    inventory = inspect_inputs(root, pairs)
    if args.mode == "inventory":
        return
    if args.closed_epochs < 1 or args.open_epochs < 1 or args.batch_size < 6 or args.supcon_batch_size < 6 or not args.seeds:
        raise RuntimeError("Invalid training parameters")
    output_root = args.output_root.expanduser().absolute()
    safe_parent = (project_root / "artifacts" / "modern_validation").resolve()
    resolved_output = output_root.resolve()
    if safe_parent not in resolved_output.parents:
        raise RuntimeError("Output root must be inside this repository's artifacts/modern_validation/ directory")
    if resolved_output == root or root in resolved_output.parents:
        raise RuntimeError("Output path must be separate from original raw datasets")
    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise RuntimeError("Run name must contain only letters, numbers, underscores and hyphens")
    output = output_root / run_name
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "inventory.csv", inventory)
    all_records, audits = [], []
    for family, capture, report in pairs:
        records, audit = extract_capture(family, capture, report, root)
        all_records.extend(records)
        audits.append(audit)
        print(f"EXTRACT {family}/{capture.name}: {len(records)} labelled positive bidirectional flows", flush=True)
    write_csv(output / "label_matching_audit.csv", audits)
    if not all_records:
        raise RuntimeError("No flows passed label/PCAP matching checks; inspect audit CSV")
    chosen, split_audit = capture_splits(all_records, SEED)
    write_csv(output / "split_manifest.csv", [{k: v for k, v in r.items() if k not in ("sizes", "iats")} for r in chosen])
    (output / "split_audit.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({"dataset": "ECU-MALNETT V2 APT subset", "doi": "10.25958/t81p-ap36",
         "provider": "Edith Cowan University", "label_type": "Construct-inferred malicious flags",
         "feature_channels": ["signed_frame_length", "log1p_interarrival_seconds"],
         "methodological_difference": "Conservative bidirectional merging of Construct directional labels; remove flows with >120s gaps, mismatched counts, or ambiguous label keys; full and five-packet fingerprint deduplication",
         "random_split_seed": SEED, "input_files": len(pairs), "pre_split_positive_flows": len(all_records),
         "post_dedup_positive_flows": len(chosen), "parameters": vars(args) | {"dataset_root": str(root), "output_root": str(output_root)}}, indent=2), encoding="utf-8")
    run_experiments(chosen, args, output)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)


if __name__ == "__main__":
    main()
