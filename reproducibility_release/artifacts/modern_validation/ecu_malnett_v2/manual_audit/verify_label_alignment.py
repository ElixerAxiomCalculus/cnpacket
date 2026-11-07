"""Read-only, record-by-record CSV/PCAP alignment audit; no experiment modules imported.

Writes new evidence files only beside this script, using exclusive creation.
The original review CSV and all prior experiment files remain unchanged.
"""

import csv
import hashlib
import ipaddress
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RAW = ROOT / "data/raw/APT"
SOURCE = ROOT / "artifacts/modern_validation/ecu_malnett_v2/apt_v2_followup_20261107/label_review_samples.csv"
MANIFEST = ROOT / "artifacts/modern_validation/ecu_malnett_v2/apt_v2_full_20261106/split_manifest.csv"
MICROSECOND = Decimal("0.000001")
IST = timezone(timedelta(hours=5, minutes=30), "IST")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def write_csv(name, rows, fields):
    with (OUT / name).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    with (OUT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def direction(row):
    src = str(ipaddress.ip_address(row["src"].strip()))
    dst = str(ipaddress.ip_address(row["dst"].strip()))
    sport, dport = int(row["sport"]), int(row["dport"])
    assert 0 <= sport <= 65535 and 0 <= dport <= 65535
    proto = row["transport"].strip().upper()
    assert proto in {"TCP", "UDP"}
    return src, dst, str(sport), str(dport), proto


def reverse(key):
    return key[1], key[0], key[3], key[2], key[4]


def canonical(key):
    return min(key, reverse(key))


def seconds(microseconds):
    return f"{microseconds // 1000000}.{microseconds % 1000000:06d}"


def tuple_string(key):
    return f"{key[0]}:{key[2]} -> {key[1]}:{key[3]} {key[4]}"


def inspect_pcap(path, wanted):
    """Independently decode classic microsecond PCAP + Ethernet/IPv4 TCP/UDP.

    Read every frame. Retain only requested five-tuples, including reverse
    directions, without inferring flow labels or importing research code.
    """
    found = []
    stats = Counter()
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data[:4] == bytes.fromhex("d4c3b2a1"):
            endian = "<"
        elif data[:4] == bytes.fromhex("a1b2c3d4"):
            endian = ">"
        else:
            raise ValueError("Unsupported PCAP magic or timestamp resolution")
        major, minor, _, _, snaplen, network = struct.unpack_from(endian + "HHiIII", data, 4)
        if (major, minor, network) != (2, 4, 1):
            raise ValueError(f"Unsupported PCAP version/link type: {major}.{minor}/{network}")
        position, first_stamp, frame_number = 24, None, 0
        while position < len(data):
            if position + 16 > len(data):
                raise ValueError("Truncated PCAP record header")
            sec, us, caplen, wirelen = struct.unpack_from(endian + "IIII", data, position)
            position += 16
            end = position + caplen
            if us >= 1000000 or end > len(data) or caplen > snaplen or caplen > wirelen:
                raise ValueError("Invalid/truncated PCAP packet record")
            stamp = sec * 1000000 + us
            if first_stamp is None:
                first_stamp = stamp
            frame_number += 1
            stats["total_frames"] += 1
            if caplen < 14:
                stats["short_ethernet"] += 1
                position = end
                continue
            ether_type = struct.unpack_from("!H", data, position + 12)[0]
            ip_pos = position + 14
            while ether_type in {0x8100, 0x88A8, 0x9100}:
                if ip_pos + 4 > end:
                    raise ValueError("Truncated VLAN header")
                ether_type = struct.unpack_from("!H", data, ip_pos + 2)[0]
                ip_pos += 4
            if ether_type != 0x0800:
                stats["non_ipv4_frames"] += 1
                position = end
                continue
            if ip_pos + 20 > end:
                raise ValueError("Truncated IPv4 header")
            version_ihl = data[ip_pos]
            ihl = (version_ihl & 15) * 4
            if version_ihl >> 4 != 4 or ihl < 20 or ip_pos + ihl > end:
                raise ValueError("Invalid IPv4 header")
            fragment = struct.unpack_from("!H", data, ip_pos + 6)[0]
            proto_number = data[ip_pos + 9]
            src = ".".join(str(x) for x in data[ip_pos + 12:ip_pos + 16])
            dst = ".".join(str(x) for x in data[ip_pos + 16:ip_pos + 20])
            if fragment & 0x1FFF:
                stats["noninitial_ipv4_fragments"] += 1
                # Port identification is unavailable for noninitial fragments.
                if any(src == k[0] and dst == k[1] and proto_number == (6 if k[4] == "TCP" else 17) for k in wanted):
                    raise ValueError("Noninitial IPv4 fragment shares a reviewed endpoint/protocol pair")
                position = end
                continue
            transport_pos = ip_pos + ihl
            if proto_number not in {6, 17}:
                position = end
                continue
            if transport_pos + 4 > end:
                raise ValueError("Truncated transport ports")
            sport, dport = struct.unpack_from("!HH", data, transport_pos)
            key = src, dst, str(sport), str(dport), "TCP" if proto_number == 6 else "UDP"
            if key in wanted:
                if fragment & 0x2000:
                    raise ValueError("Reviewed flow contains fragmented IPv4 datagrams")
                found.append({"frame_number": frame_number, "timestamp_us": stamp,
                              "relative_us": stamp - first_stamp, "wire_length": wirelen,
                              "captured_length": caplen, "key": key})
            position = end
        stats["first_frame_timestamp_us"] = first_stamp
        stats["pcap_bytes_consumed"] = position
    return found, dict(stats)


def crosscheck_tshark(path, wanted, raw_packets):
    conditions = []
    for src, dst, sport, dport, transport in sorted(wanted):
        layer = transport.lower()
        conditions.append(f"(ip.src == {src} && ip.dst == {dst} && {layer}.srcport == {sport} && {layer}.dstport == {dport})")
    fields = ["frame.number", "frame.time_epoch", "frame.time_relative", "frame.len", "frame.cap_len",
              "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "_ws.col.Protocol"]
    command = ["tshark", "-n", "-r", str(path), "-Y", " || ".join(conditions),
               "-T", "fields", "-E", "occurrence=f"]
    for field in fields:
        command += ["-e", field]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    observed = []
    if result.returncode:
        return False, [], command, result.stderr.strip()
    for line in result.stdout.splitlines():
        values = line.split("\t")
        if len(values) != len(fields):
            return False, observed, command, "Unexpected TShark field count"
        number, stamp, relative, wire, caplen, src, dst, ts, td, us, ud, protocol = values
        key = (src, dst, ts, td, "TCP") if ts and td else (src, dst, us, ud, "UDP")
        observed.append({"frame_number": int(number), "timestamp_us": int(Decimal(stamp) * 1000000),
                         "relative_us": int(Decimal(relative) * 1000000), "wire_length": int(wire),
                         "captured_length": int(caplen), "key": key, "dissected_protocol": protocol})
    same = len(raw_packets) == len(observed) and all(
        all(left[k] == right[k] for k in left) for left, right in zip(raw_packets, observed)
    )
    return same, observed, command, result.stderr.strip()


def protected_metadata():
    """Snapshot existing datasets, research source and results; exclude new output."""
    snapshot = {}
    for directory in (ROOT / "data", ROOT / "scripts", ROOT / "controller", ROOT / "models", ROOT / "artifacts"):
        for base, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}
                       and not (Path(base) / d).resolve().is_relative_to(OUT)]
            for name in files:
                path = Path(base) / name
                details = path.lstat()
                snapshot[str(path.relative_to(ROOT))] = [details.st_size, details.st_mtime_ns, details.st_mode]
    return snapshot


def git_state():
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    prefix = str(OUT.relative_to(ROOT)) + "/"
    status = subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, env=env).decode()
    filtered = "".join(x + "\n" for x in status.splitlines() if not x[3:].startswith(prefix))
    diff = subprocess.check_output(["git", "--no-optional-locks", "diff", "--binary", "HEAD"], cwd=ROOT, env=env)
    return {"status_excluding_new_audit": filtered, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def main():
    outputs = ["label_review_samples_verified.csv", "directional_evidence.csv", "packet_evidence.csv",
               "audit_evidence.json", "audit_report.md", "source_integrity.json"]
    if any((OUT / name).exists() for name in outputs):
        raise RuntimeError("Audit output exists; refusing to overwrite")
    started = datetime.now(IST).isoformat()
    before_metadata, before_git = protected_metadata(), git_state()
    samples, original_columns = read_csv(SOURCE)
    assert len(samples) == 25
    manifest, _ = read_csv(MANIFEST)
    frozen = defaultdict(list)
    for row in manifest:
        frozen[(row["family"], row["capture"], row["flow_id"])].append(row)
    grouped = defaultdict(list)
    csv_data, protected_inputs = {}, {SOURCE, MANIFEST}
    for row in samples:
        capture = RAW / row["capture"]
        report = RAW / f"analysis_output_{row['family'].lower()}_baseline_labelled" / f"{capture.stem}_sessions.csv"
        assert capture.is_file() and report.is_file() and capture.resolve().is_relative_to(RAW) and report.resolve().is_relative_to(RAW)
        protected_inputs.update((capture, report))
        if report not in csv_data:
            csv_data[report] = read_csv(report)[0]
        grouped[capture].append((row, report))
    protected_inputs.update((ROOT / "scripts/modern_validation").glob("*.py"))
    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(protected_inputs)}
    print(f"Read-only source snapshot: {len(before_hashes)} files; {len(grouped)} original captures", flush=True)
    flow_results, capture_results, packet_rows, directional_rows = [], [], [], []
    for capture_number, (capture, group) in enumerate(grouped.items(), 1):
        wanted = set()
        for sample, report in group:
            for record_number in map(int, sample["construct_positive_csv_record_numbers"].split(";")):
                key = direction(csv_data[report][record_number - 2])
                wanted.update((key, reverse(key)))
        error = None
        try:
            packets, capture_stats = inspect_pcap(capture, wanted)
            decoder_agreement, tshark_rows, command, warning = crosscheck_tshark(capture, wanted, packets)
        except (ValueError, OSError, struct.error) as exc:
            packets, tshark_rows, capture_stats, command, warning = [], [], {}, [], ""
            decoder_agreement, error = False, str(exc)
        by_direction = defaultdict(list)
        for packet in packets:
            by_direction[packet["key"]].append(packet)
        protocols_by_frame = {p["frame_number"]: p["dissected_protocol"] for p in tshark_rows}
        capture_results.append({"capture": str(capture.relative_to(RAW)), "independent_decode": capture_stats,
                                "tshark_command": command, "decoder_agreement": decoder_agreement,
                                "tshark_stderr": warning, "error": error})
        for sample, report in group:
            index = samples.index(sample) + 1
            numbers = [int(n) for n in sample["construct_positive_csv_record_numbers"].split(";")]
            supplied_rows = [csv_data[report][n - 2] for n in numbers]
            selected_key = canonical(direction(supplied_rows[0]))
            flow_packets = [p for p in packets if canonical(p["key"]) == selected_key]
            problems, mismatches = [], []
            if not decoder_agreement:
                problems.append("Independent PCAP/TShark comparison failed" + (": " + error if error else ""))
            calculated_id = hashlib.sha256((sample["capture"] + repr(selected_key)).encode()).hexdigest()[:24]
            if calculated_id != sample["flow_id"]:
                mismatches.append("flow_id does not identify the Construct five-tuple")
            earlier = frozen[(sample["family"], sample["capture"], sample["flow_id"])]
            if len(earlier) != 1 or earlier[0]["split"] != sample["split"] or earlier[0]["packet_count"] != sample["pcap_bidir_packet_count"]:
                mismatches.append("Review metadata differs from the frozen manifest")
            if len(flow_packets) != int(sample["pcap_bidir_packet_count"]):
                mismatches.append(f"Bidirectional count {len(flow_packets)} != supplied {sample['pcap_bidir_packet_count']}")
            tuple_rows = []
            for record_number, row in enumerate(csv_data[report], 2):
                try:
                    key = direction(row)
                except (ValueError, AssertionError):
                    continue
                if canonical(key) == selected_key:
                    tuple_rows.append((record_number, row, key))
            actual_positive = [n for n, row, _ in tuple_rows if row.get("malicious", "").strip() == "1"]
            if actual_positive != numbers:
                mismatches.append("Listed positive Construct record numbers differ from original CSV")
            if len({key for _, _, key in tuple_rows}) != len(tuple_rows):
                problems.append("Duplicate Construct five-tuple: row-to-packet correspondence ambiguous")
            if ";".join(row["packet_count"] for row in supplied_rows) != sample["construct_directional_packet_counts"]:
                mismatches.append("Directional counts differ from the original Construct rows")
            if ";".join(row["score"] for row in supplied_rows) != sample["construct_scores"]:
                mismatches.append("Scores differ from the original Construct rows")
            if " || ".join(row["reasons"] for row in supplied_rows)[:1200] != sample["construct_reasons"]:
                mismatches.append("Reasons differ from the original Construct rows")
            details = []
            for record_number, row, key in tuple_rows:
                observed = by_direction[key]
                if len(observed) != int(row["packet_count"]):
                    mismatches.append(f"CSV record {record_number}: directional count differs")
                first = observed[0]["timestamp_us"] if observed else None
                last = observed[-1]["timestamp_us"] if observed else None
                timestamp_delta = Decimal(seconds(last)) - Decimal(row["timestamp"]) if last is not None else None
                duration_delta = Decimal(seconds(last - first)) - Decimal(row["session_duration"]) if observed and last >= first else None
                if timestamp_delta is None or abs(timestamp_delta) > MICROSECOND:
                    mismatches.append(f"CSV record {record_number}: timestamp does not match last packet")
                if duration_delta is None or abs(duration_delta) > MICROSECOND:
                    mismatches.append(f"CSV record {record_number}: duration does not match first/last packets")
                labels = sorted({protocols_by_frame.get(p["frame_number"], "") for p in observed})
                csv_protocol = row["protocol"].upper()
                app_check = (csv_protocol in {"TCP", "UDP"} and csv_protocol == key[4]) or (
                    csv_protocol == "TLS" and any("TLS" in p.upper() or "SSL" in p.upper() for p in labels)) or (
                    csv_protocol in {"DNS", "QUIC"} and any(csv_protocol in p.upper() for p in labels))
                if not app_check:
                    problems.append(f"CSV record {record_number}: application protocol {row['protocol']} not corroborated by packet dissection")
                evidence = {"sample_number": index, "family": sample["family"], "capture": sample["capture"],
                            "flow_id": sample["flow_id"], "construct_csv": str(report.relative_to(ROOT)),
                            "construct_csv_record_number": record_number, "construct_flow_id": row["flow_id"],
                            "src_ip": key[0], "dst_ip": key[1], "src_port": key[2], "dst_port": key[3],
                            "transport": key[4], "construct_protocol": row["protocol"], "tshark_protocols": ";".join(labels),
                            "construct_packet_count": row["packet_count"], "pcap_directional_count": len(observed),
                            "construct_timestamp_s": row["timestamp"], "pcap_first_timestamp_s": seconds(first) if first is not None else "",
                            "pcap_last_timestamp_s": seconds(last) if last is not None else "", "last_timestamp_delta_s": str(timestamp_delta),
                            "construct_session_duration_s": row["session_duration"], "duration_delta_s": str(duration_delta),
                            "construct_malicious_flag": row["malicious"], "construct_score": row["score"], "construct_reasons": row["reasons"],
                            "packet_frame_numbers": ";".join(str(p["frame_number"]) for p in observed),
                            "independent_decoders_agree": decoder_agreement, "application_protocol_corroborated": app_check}
                directional_rows.append(evidence)
                details.append({"csv_record": record_number, "tuple": list(key), "count": len(observed),
                                "first_timestamp_s": evidence["pcap_first_timestamp_s"], "last_timestamp_s": evidence["pcap_last_timestamp_s"],
                                "timestamp_delta_s": str(timestamp_delta), "duration_delta_s": str(duration_delta)})
            known_directions = {key for _, _, key in tuple_rows}
            if any(p["key"] not in known_directions for p in flow_packets):
                problems.append("Observed reverse-direction packets lack a corresponding Construct row")
            times = [p["timestamp_us"] for p in flow_packets]
            max_gap = max((b - a for a, b in zip(times, times[1:])), default=0)
            if any(b < a for a, b in zip(times, times[1:])) or max_gap > 120000000:
                problems.append("Five-tuple has time-order/idle-gap ambiguity")
            problems += mismatches
            status = "unresolved" if problems else "checked"
            summary = "; ".join(f"CSV {d['csv_record']}: {tuple_string(tuple(d['tuple']))}, n={d['count']}, first={d['first_timestamp_s']}s, last={d['last_timestamp_s']}s" for d in details)
            notes = ("Verified CSV-to-PCAP alignment with independent header decoder and TShark; " if status == "checked" else "UNRESOLVED: " + " | ".join(problems) + "; ")
            notes += summary + ". Stored PCAP timestamp equals Construct terminal timestamp; duration checked within 1 us. Alignment only; heuristic Construct labels do not independently prove malicious activity."
            if status != "checked":
                notes = "UNRESOLVED: " + " | ".join(problems) + "; " + summary + ". Alignment not fully verified; heuristic labels do not prove malicious activity."
            flow_results.append({"sample_number": index, "family": sample["family"], "capture": sample["capture"],
                                 "flow_id": sample["flow_id"], "status": status, "mismatch_detected": bool(mismatches),
                                 "problems": problems, "directions": details, "observed_bidir_packet_count": len(flow_packets),
                                 "max_interpacket_gap_s": seconds(max_gap), "review_notes": notes})
            for packet in flow_packets:
                key = packet["key"]
                packet_rows.append({"sample_number": index, "flow_id": sample["flow_id"], "capture": sample["capture"],
                                    "frame_number": packet["frame_number"], "timestamp_s": seconds(packet["timestamp_us"]),
                                    "frame_time_relative_s": seconds(packet["relative_us"]), "src_ip": key[0], "dst_ip": key[1],
                                    "src_port": key[2], "dst_port": key[3], "transport": key[4],
                                    "frame_length": packet["wire_length"], "captured_length": packet["captured_length"],
                                    "tshark_protocol": protocols_by_frame.get(packet["frame_number"], "")})
        print(f"Capture {capture_number}/{len(grouped)}: {capture.parent.name}/{capture.stem[:12]}; selected packets={len(packets)}; independent/TShark agreement={decoder_agreement}", flush=True)
    flow_results.sort(key=lambda row: row["sample_number"])
    verified_rows = []
    for original_row, result in zip(samples, flow_results):
        updated = dict(original_row, review_status=result["status"], review_notes=result["review_notes"])
        assert all(updated[k] == original_row[k] for k in original_columns if k not in {"review_status", "review_notes"})
        verified_rows.append(updated)
    after_hashes = {name: sha256(ROOT / name) for name in before_hashes}
    changed_hashes = [name for name in before_hashes if before_hashes[name] != after_hashes[name]]
    after_metadata, after_git = protected_metadata(), git_state()
    changed_metadata = [name for name in sorted(before_metadata.keys() | after_metadata.keys()) if before_metadata.get(name) != after_metadata.get(name)]
    if changed_hashes or changed_metadata or before_git != after_git:
        raise RuntimeError(f"Protected originals changed: content={changed_hashes}, metadata={changed_metadata}, git={before_git != after_git}; no restoration attempted")
    counts = Counter(row["status"] for row in flow_results)
    mismatched = sum(row["mismatch_detected"] for row in flow_results)
    counts_summary = {"verified": counts["checked"], "unresolved": counts["unresolved"], "mismatched": mismatched,
                      "mismatches_are_a_subset_of_unresolved": True, "total_review_records": len(samples)}
    write_csv("label_review_samples_verified.csv", verified_rows, original_columns)
    write_csv("directional_evidence.csv", sorted(directional_rows, key=lambda row: (row["sample_number"], row["construct_csv_record_number"])), list(directional_rows[0]))
    write_csv("packet_evidence.csv", sorted(packet_rows, key=lambda row: (row["sample_number"], row["frame_number"])), list(packet_rows[0]))
    tshark_version = subprocess.check_output(["tshark", "--version"], text=True).splitlines()[0]
    write_json("audit_evidence.json", {"started_at_Asia_Kolkata": started, "completed_at_Asia_Kolkata": datetime.now(IST).isoformat(),
                                      "counts": counts_summary, "timestamp_tolerance_s": str(MICROSECOND), "tshark_version": tshark_version,
                                      "source_review_csv": str(SOURCE.relative_to(ROOT)), "capture_count": len(grouped),
                                      "construct_directional_rows": len(directional_rows), "packet_evidence_rows": len(packet_rows),
                                      "flow_results": flow_results, "capture_results": capture_results,
                                      "scope": "Assistant tool-assisted record-by-record alignment verification; not an external human sign-off or malware ground-truth certification."})
    write_json("source_integrity.json", {"source_sha256_before": before_hashes, "source_sha256_after": after_hashes,
                                        "changed_source_hashes": changed_hashes, "protected_metadata_file_count": len(before_metadata),
                                        "changed_original_metadata_paths": changed_metadata, "git_before": before_git, "git_after": after_git,
                                        "writes_restricted_to": str(OUT.relative_to(ROOT)),
                                        "scope": "Relevant original PCAPs/Construct CSVs, original review CSV/frozen manifest and modern-validation source content hashes; all data/scripts/controller/models/artifacts metadata except new audit directory and Git/cache subtrees. Access times and directory mtimes are excluded."})
    lines = ["# Phase 1 manual label alignment audit", "", f"Date: {started}. Reviewer: assistant, using tool-assisted record-by-record inspection; no external human sign-off is claimed.", "",
             f"**Verified: {counts['checked']}/25. Unresolved: {counts['unresolved']}/25. Mismatched: {mismatched}/25.** Mismatched means an observed numerical/identity discrepancy and is a subset of unresolved, not a third disjoint status.", "",
             "## Sources and verification", "", f"Read `{SOURCE.relative_to(ROOT)}` and the original Construct sessions CSVs/PCAPs in `data/raw/APT/`: {len(grouped)} captures, {len(directional_rows)} directional Construct rows, {len(packet_rows)} reviewed packets. Each listed CSV record number is the 1-based logical CSV record including header record 1; it is not the Construct `flow_N` identifier. No cached extracted sequences or prior matching verdicts were used as packet evidence.", "",
             "A separate standard-library decoder read every original Ethernet/IPv4 PCAP frame and selected both directions of each exact IP/port/TCP-or-UDP tuple. Fresh TShark exact-tuple display filters cross-checked every selected frame number, stored timestamp, frame-relative timestamp, captured/wire length and tuple. CSV directional counts and the supplied bidirectional counts were compared to all observed packets. Original positive-row numbers, scores/reasons, canonical flow-ID derivation and frozen-manifest identity/count/split were checked without regenerating or changing splits. Full commands and per-flow findings are recorded in `audit_evidence.json`.", "",
             "**Timestamp interpretation:** Construct `timestamp` is checked against the final directional packet's stored PCAP timestamp. `session_duration` is checked against last minus first. All captures use microsecond PCAP timestamps close to origin zero; these are not independently established wall-clock dates. TShark `frame.time_relative` subtracts the capture's first-frame timestamp (10–16 microseconds here), so blindly comparing that field to Construct `timestamp` would introduce an offset. No fitted per-flow offset was applied. The prespecified numerical tolerance is one microsecond, matching the PCAP timestamp resolution. Actual residuals are retained in `directional_evidence.csv`.", "",
             "**Protocol interpretation:** TCP/UDP is verified from IP protocol and transport ports. Construct `protocol` is an application/dissector summary for TLS, QUIC or DNS in three rows; those are corroborated with packet dissection. Generic TCP summaries do not require every packet's application display label to be TCP. The two bidirectional samples (7 and 12) are checked as two separate directional Construct rows whose counts sum to the full observed flow.", "",
             "## Record outcomes", "", "| Sample | Family | Flow ID | Observed packets | Result |", "|---:|---|---|---:|---|"]
    lines += [f"| {x['sample_number']} | {x['family']} | `{x['flow_id']}` | {x['observed_bidir_packet_count']} | {x['status']}{' (mismatch)' if x['mismatch_detected'] else ''} |" for x in flow_results]
    lines += ["", "## Outputs and limits", "", "- `label_review_samples_verified.csv`: all 25 original rows and columns; only `review_status`/`review_notes` are populated in this independent copy.",
              "- `directional_evidence.csv`: original CSV references, exact directional endpoints/protocols/counts, first/last timestamps, timing residuals and packet frame numbers.",
              "- `packet_evidence.csv`: independent per-packet tuple/timing evidence and TShark protocol labels; no payload bytes are exported.",
              "- `audit_evidence.json`: record-level decisions, parser checks, commands and capture diagnostics.",
              "- `source_integrity.json`: before/after SHA-256 checks and original-file/Git comparisons.",
              "- `verify_label_alignment.py`: new read-only audit implementation; it refuses to overwrite audit outputs.", "",
              f"Original integrity: {len(before_hashes)} relevant input/source files were SHA-256 checked before/after; {len(before_metadata)} original data/source/model/result file metadata entries were unchanged; pre-existing Git status and tracked diff were unchanged. The original review CSV, datasets, experiment source, frozen splits, checkpoints and existing results were not modified. No training, inference, experiment rerun or GitHub update was performed.", "",
              "**Scope limitation:** `checked` verifies CSV-to-PCAP identity, counts and timing only. The retained `malicious=1`, score 5 and reasons are Construct heuristics, not independently established malicious activity or family attribution. Most reviewed reasons are unexplained traffic outside a baseline/whitelist; sample 15 is an out-of-baseline DNS reason, and samples 7/12 include inherited paired-flow labeling. Endpoint/port/protocol alignment cannot establish whether this traffic is benign, malicious, command-and-control, or generated by the named family. These 25 selected records do not validate all dataset labels or authorize bypassing a human-review gate in the original workflow.", ""]
    with (OUT / "audit_report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps(counts_summary), flush=True)


if __name__ == "__main__":
    main()
