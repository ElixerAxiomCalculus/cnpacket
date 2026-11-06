import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "phase2"
MANIFEST = ARTIFACTS / "iscxvpn_protocol_manifest.csv"
RAW = ROOT / "data" / "raw" / "iscxvpn2016"

FLOW_IDLE_TIMEOUT_SECONDS = 120.0
MAX_SEQUENCE_PACKETS = 50
WRITE_BATCH_SIZE = 10_000

SCHEMA = pa.schema(
    [
        ("domain", pa.string()),
        ("activity", pa.string()),
        ("fold", pa.int8()),
        ("archive", pa.string()),
        ("capture", pa.string()),
        ("flow_id", pa.string()),
        ("protocol", pa.string()),
        ("endpoint_a_ip", pa.string()),
        ("endpoint_a_port", pa.int32()),
        ("endpoint_b_ip", pa.string()),
        ("endpoint_b_port", pa.int32()),
        ("segment", pa.int64()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration_seconds", pa.float64()),
        ("packet_count", pa.int64()),
        ("byte_count", pa.int64()),
        ("sequence_packets", pa.int32()),
        ("signed_packet_sizes", pa.list_(pa.int32())),
        ("interarrival_seconds", pa.list_(pa.float64())),
        ("valid_ge3", pa.bool_()),
        ("valid_ge5", pa.bool_()),
        ("valid_ge10", pa.bool_()),
        ("valid_ge20", pa.bool_()),
        ("valid_ge50", pa.bool_())
    ]
)


@dataclass
class FlowState:
    protocol: str
    endpoint_a_ip: str
    endpoint_a_port: int
    endpoint_b_ip: str
    endpoint_b_port: int
    segment: int
    start_time: float
    last_time: float
    packet_count: int = 0
    byte_count: int = 0
    signed_packet_sizes: list = field(default_factory=list)
    interarrival_seconds: list = field(default_factory=list)


class ParquetSink:
    def __init__(self, path):
        self.path = path
        self.writer = None
        self.buffer = []

    def append(self, row):
        self.buffer.append(row)

        if len(self.buffer) >= WRITE_BATCH_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        table = pa.Table.from_pylist(
            self.buffer,
            schema=SCHEMA
        )

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                SCHEMA,
                compression="zstd"
            )

        self.writer.write_table(table)
        self.buffer.clear()

    def close(self):
        self.flush()

        if self.writer is not None:
            self.writer.close()
        else:
            pq.write_table(
                pa.Table.from_pylist(
                    [],
                    schema=SCHEMA
                ),
                self.path,
                compression="zstd"
            )


def sanitize(value):
    return "".join(
        character
        if character.isalnum()
        or character in {"-", "_"}
        else "_"
        for character in value
    )


def output_basename(row):
    identity = (
        f"{row['archive']}|"
        f"{row['capture']}"
    )

    suffix = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{sanitize(row['domain'])}__"
        f"{sanitize(row['activity'])}__"
        f"{sanitize(Path(row['capture']).stem)}__"
        f"{suffix}"
    )


def canonical_flow_key(
    protocol,
    src_ip,
    src_port,
    dst_ip,
    dst_port
):
    source = (
        src_ip,
        src_port
    )

    destination = (
        dst_ip,
        dst_port
    )

    if source <= destination:
        endpoint_1 = source
        endpoint_2 = destination
    else:
        endpoint_1 = destination
        endpoint_2 = source

    return (
        protocol,
        endpoint_1[0],
        endpoint_1[1],
        endpoint_2[0],
        endpoint_2[1]
    )


def stable_flow_id(
    archive_name,
    capture_name,
    state
):
    text = (
        f"{archive_name}|"
        f"{capture_name}|"
        f"{state.protocol}|"
        f"{state.endpoint_a_ip}:"
        f"{state.endpoint_a_port}|"
        f"{state.endpoint_b_ip}:"
        f"{state.endpoint_b_port}|"
        f"{state.segment}"
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:24]


def state_to_row(
    state,
    manifest_row
):
    packet_count = state.packet_count

    return {
        "domain": manifest_row["domain"],
        "activity": manifest_row["activity"],
        "fold": int(manifest_row["fold"]),
        "archive": manifest_row["archive"],
        "capture": manifest_row["capture"],
        "flow_id": stable_flow_id(
            manifest_row["archive"],
            manifest_row["capture"],
            state
        ),
        "protocol": state.protocol,
        "endpoint_a_ip": state.endpoint_a_ip,
        "endpoint_a_port": state.endpoint_a_port,
        "endpoint_b_ip": state.endpoint_b_ip,
        "endpoint_b_port": state.endpoint_b_port,
        "segment": state.segment,
        "start_time": state.start_time,
        "end_time": state.last_time,
        "duration_seconds": max(
            0.0,
            state.last_time - state.start_time
        ),
        "packet_count": packet_count,
        "byte_count": state.byte_count,
        "sequence_packets": len(
            state.signed_packet_sizes
        ),
        "signed_packet_sizes": state.signed_packet_sizes,
        "interarrival_seconds": state.interarrival_seconds,
        "valid_ge3": packet_count >= 3,
        "valid_ge5": packet_count >= 5,
        "valid_ge10": packet_count >= 10,
        "valid_ge20": packet_count >= 20,
        "valid_ge50": packet_count >= 50
    }


def find_seven_zip():
    executable = (
        shutil.which("7zz")
        or shutil.which("7z")
    )

    if executable is None:
        raise RuntimeError(
            "Neither 7zz nor 7z was found. "
            "Install the 7zip command-line package."
        )

    return executable


def materialize_capture(
    manifest_row,
    temp_directory
):
    archive_path = (
        RAW
        / manifest_row["archive"]
    )

    if not archive_path.exists():
        raise RuntimeError(
            f"Missing archive: {archive_path}"
        )

    if temp_directory.exists():
        shutil.rmtree(
            temp_directory
        )

    temp_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    member_name = (
        manifest_row["capture"]
    )

    destination = (
        temp_directory
        / Path(member_name).name
    )

    with zipfile.ZipFile(
        archive_path,
        "r"
    ) as archive:
        try:
            info = archive.getinfo(
                member_name
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Missing member "
                f"{member_name} "
                f"in {archive_path.name}"
            ) from exc

        expected_size = (
            info.file_size
        )

    manifest_size = int(
        manifest_row[
            "uncompressed_bytes"
        ]
    )

    if expected_size != manifest_size:
        raise RuntimeError(
            f"ZIP/manifest size mismatch for "
            f"{member_name}: "
            f"{expected_size} != {manifest_size}"
        )

    executable = find_seven_zip()

    with destination.open(
        "wb"
    ) as target:
        result = subprocess.run(
            [
                executable,
                "x",
                "-so",
                str(archive_path),
                member_name
            ],
            stdout=target,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"7-Zip failed to extract "
            f"{member_name} from "
            f"{archive_path.name} with code "
            f"{result.returncode}: "
            f"{result.stderr[-4000:]}"
        )

    actual_size = destination.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"Extracted size mismatch for "
            f"{member_name}: "
            f"{actual_size} != {expected_size}"
        )

    return destination


def process_capture(
    manifest_row,
    output_directory,
    stats_directory,
    log_directory,
    temp_directory
):
    capture_path = materialize_capture(
        manifest_row,
        temp_directory
    )

    base = output_basename(
        manifest_row
    )

    final_output = (
        output_directory
        / f"{base}.parquet"
    )

    partial_output = (
        output_directory
        / f"{base}.parquet.partial"
    )

    stats_path = (
        stats_directory
        / f"{base}.json"
    )

    stderr_path = (
        log_directory
        / f"{base}.tshark.log"
    )

    if partial_output.exists():
        partial_output.unlink()

    command = [
        "tshark",
        "-n",
        "-r",
        str(capture_path),
        "-Y",
        "(tcp || udp) && (ip || ipv6)",
        "-T",
        "fields",
        "-E",
        "header=n",
        "-E",
        "quote=n",
        "-E",
        "occurrence=f",
        "-E",
        "separator=\t",
        "-e",
        "frame.number",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "ipv6.src",
        "-e",
        "ipv6.dst",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport"
    ]

    active = OrderedDict()

    eligible_packets = 0
    skipped_packets = 0
    total_flows = 0

    counts = {
        3: 0,
        5: 0,
        10: 0,
        20: 0,
        50: 0
    }

    sink = ParquetSink(
        partial_output
    )

    def finalize(state):
        nonlocal total_flows

        row = state_to_row(
            state,
            manifest_row
        )

        sink.append(row)
        total_flows += 1

        for threshold in counts:
            if state.packet_count >= threshold:
                counts[threshold] += 1

    try:
        with stderr_path.open(
            "w",
            encoding="utf-8"
        ) as error_handle:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=error_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            if process.stdout is None:
                raise RuntimeError(
                    "Unable to read TShark stdout"
                )

            for line in process.stdout:
                fields = line.rstrip(
                    "\r\n"
                ).split("\t")

                if len(fields) != 11:
                    skipped_packets += 1
                    continue

                (
                    frame_number,
                    timestamp_text,
                    frame_length_text,
                    ipv4_source,
                    ipv4_destination,
                    ipv6_source,
                    ipv6_destination,
                    tcp_source_port,
                    tcp_destination_port,
                    udp_source_port,
                    udp_destination_port
                ) = fields

                try:
                    timestamp = float(
                        timestamp_text
                    )

                    frame_length = int(
                        frame_length_text
                    )

                    source_ip = (
                        ipv4_source
                        or ipv6_source
                    )

                    destination_ip = (
                        ipv4_destination
                        or ipv6_destination
                    )

                    if (
                        tcp_source_port
                        and tcp_destination_port
                    ):
                        protocol = "TCP"
                        source_port = int(
                            tcp_source_port
                        )
                        destination_port = int(
                            tcp_destination_port
                        )

                    elif (
                        udp_source_port
                        and udp_destination_port
                    ):
                        protocol = "UDP"
                        source_port = int(
                            udp_source_port
                        )
                        destination_port = int(
                            udp_destination_port
                        )

                    else:
                        skipped_packets += 1
                        continue

                    if (
                        not source_ip
                        or not destination_ip
                    ):
                        skipped_packets += 1
                        continue

                except ValueError:
                    skipped_packets += 1
                    continue

                while active:
                    oldest_key, oldest_state = next(
                        iter(
                            active.items()
                        )
                    )

                    if (
                        timestamp
                        - oldest_state.last_time
                        <= FLOW_IDLE_TIMEOUT_SECONDS
                    ):
                        break

                    finalize(
                        oldest_state
                    )

                    del active[
                        oldest_key
                    ]

                eligible_packets += 1

                base_key = canonical_flow_key(
                    protocol,
                    source_ip,
                    source_port,
                    destination_ip,
                    destination_port
                )

                state = active.get(
                    base_key
                )

                if state is None:
                    state = FlowState(
                        protocol=protocol,
                        endpoint_a_ip=source_ip,
                        endpoint_a_port=source_port,
                        endpoint_b_ip=destination_ip,
                        endpoint_b_port=destination_port,
                        segment=int(
                            frame_number
                        ),
                        start_time=timestamp,
                        last_time=timestamp
                    )

                    active[
                        base_key
                    ] = state

                    direction = 1
                    interarrival = 0.0

                else:
                    source_endpoint = (
                        source_ip,
                        source_port
                    )

                    destination_endpoint = (
                        destination_ip,
                        destination_port
                    )

                    endpoint_a = (
                        state.endpoint_a_ip,
                        state.endpoint_a_port
                    )

                    endpoint_b = (
                        state.endpoint_b_ip,
                        state.endpoint_b_port
                    )

                    if (
                        source_endpoint == endpoint_a
                        and
                        destination_endpoint == endpoint_b
                    ):
                        direction = 1

                    elif (
                        source_endpoint == endpoint_b
                        and
                        destination_endpoint == endpoint_a
                    ):
                        direction = -1

                    else:
                        raise RuntimeError(
                            "Canonical flow-key inconsistency"
                        )

                    interarrival = max(
                        0.0,
                        timestamp - state.last_time
                    )

                state.packet_count += 1
                state.byte_count += (
                    frame_length
                )

                if (
                    len(
                        state.signed_packet_sizes
                    )
                    < MAX_SEQUENCE_PACKETS
                ):
                    state.signed_packet_sizes.append(
                        direction
                        * frame_length
                    )

                    state.interarrival_seconds.append(
                        interarrival
                    )

                state.last_time = timestamp

                active.move_to_end(
                    base_key
                )

            return_code = process.wait()

        if return_code != 0:
            message = stderr_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            raise RuntimeError(
                f"TShark failed for "
                f"{manifest_row['capture']} "
                f"with code {return_code}\n"
                f"{message[-4000:]}"
            )

        for state in active.values():
            finalize(
                state
            )

        sink.close()

        partial_output.replace(
            final_output
        )

        result = {
            "domain": manifest_row["domain"],
            "activity": manifest_row["activity"],
            "fold": int(
                manifest_row["fold"]
            ),
            "archive": manifest_row["archive"],
            "capture": manifest_row["capture"],
            "eligible_packets": eligible_packets,
            "skipped_packets": skipped_packets,
            "flows_total": total_flows,
            "flows_ge3": counts[3],
            "flows_ge5": counts[5],
            "flows_ge10": counts[10],
            "flows_ge20": counts[20],
            "flows_ge50": counts[50],
            "output_parquet": str(
                final_output.relative_to(
                    ROOT
                )
            )
        }

        stats_path.write_text(
            json.dumps(
                result,
                indent=2
            )
            + "\n"
        )

        return result

    finally:
        if partial_output.exists():
            partial_output.unlink()

        if temp_directory.exists():
            shutil.rmtree(
                temp_directory
            )


def load_primary_manifest():
    with MANIFEST.open(
        newline="",
        encoding="utf-8"
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    selected = [
        row
        for row in rows
        if row[
            "primary_selected"
        ].lower()
        == "true"
    ]

    if len(selected) != 133:
        raise RuntimeError(
            f"Expected 133 primary captures, "
            f"found {len(selected)}"
        )

    for row in selected:
        fold = int(
            row["fold"]
        )

        if fold not in {
            0,
            1,
            2,
            3,
            4
        }:
            raise RuntimeError(
                f"Invalid fold: {row}"
            )

    return selected


def generate_summaries(
    output_directory,
    stats_directory,
    artifact_prefix
):
    parquet_files = sorted(
        output_directory.glob(
            "*.parquet"
        )
    )

    if not parquet_files:
        raise RuntimeError(
            "No flow Parquet files found"
        )

    parquet_glob = str(
        output_directory
        / "*.parquet"
    ).replace(
        "'",
        "''"
    )

    connection = duckdb.connect()

    class_summary = (
        connection.execute(
            f"""
            SELECT
                domain,
                activity,
                COUNT(*) AS flows_total,
                SUM(
                    CASE
                        WHEN packet_count >= 3
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge3,
                SUM(
                    CASE
                        WHEN packet_count >= 5
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge5,
                SUM(
                    CASE
                        WHEN packet_count >= 10
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge10,
                SUM(
                    CASE
                        WHEN packet_count >= 20
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge20,
                SUM(
                    CASE
                        WHEN packet_count >= 50
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge50,
                SUM(packet_count)
                    AS eligible_packets,
                MEDIAN(packet_count)
                    AS median_packets_per_flow,
                QUANTILE_CONT(
                    packet_count,
                    0.90
                ) AS p90_packets_per_flow,
                MEDIAN(duration_seconds)
                    AS median_duration_seconds,
                QUANTILE_CONT(
                    duration_seconds,
                    0.90
                ) AS p90_duration_seconds
            FROM read_parquet(
                '{parquet_glob}'
            )
            GROUP BY
                domain,
                activity
            ORDER BY
                domain,
                activity
            """
        ).df()
    )

    fold_summary = (
        connection.execute(
            f"""
            SELECT
                fold,
                domain,
                activity,
                COUNT(*) AS flows_total,
                SUM(
                    CASE
                        WHEN packet_count >= 3
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge3,
                SUM(
                    CASE
                        WHEN packet_count >= 5
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge5,
                SUM(
                    CASE
                        WHEN packet_count >= 10
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge10,
                SUM(
                    CASE
                        WHEN packet_count >= 20
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge20,
                SUM(
                    CASE
                        WHEN packet_count >= 50
                        THEN 1 ELSE 0
                    END
                ) AS flows_ge50
            FROM read_parquet(
                '{parquet_glob}'
            )
            GROUP BY
                fold,
                domain,
                activity
            ORDER BY
                fold,
                domain,
                activity
            """
        ).df()
    )

    connection.close()

    class_path = (
        ARTIFACTS
        / f"{artifact_prefix}_class_summary.csv"
    )

    fold_path = (
        ARTIFACTS
        / f"{artifact_prefix}_fold_summary.csv"
    )

    capture_path = (
        ARTIFACTS
        / f"{artifact_prefix}_capture_summary.csv"
    )

    class_summary.to_csv(
        class_path,
        index=False
    )

    fold_summary.to_csv(
        fold_path,
        index=False
    )

    stats = []

    for path in sorted(
        stats_directory.glob(
            "*.json"
        )
    ):
        stats.append(
            json.loads(
                path.read_text()
            )
        )

    fieldnames = [
        "domain",
        "activity",
        "fold",
        "archive",
        "capture",
        "eligible_packets",
        "skipped_packets",
        "flows_total",
        "flows_ge3",
        "flows_ge5",
        "flows_ge10",
        "flows_ge20",
        "flows_ge50",
        "output_parquet"
    ]

    with capture_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in sorted(
            stats,
            key=lambda item: (
                item["domain"],
                item["activity"],
                item["archive"],
                item["capture"]
            )
        ):
            writer.writerow(
                row
            )

    return (
        class_path,
        fold_path,
        capture_path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only-capture",
        default=None
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    arguments = parser.parse_args()

    rows = load_primary_manifest()

    if arguments.only_capture:
        rows = [
            row
            for row in rows
            if row["capture"]
            == arguments.only_capture
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected exactly one match "
                f"for {arguments.only_capture}, "
                f"found {len(rows)}"
            )

    if arguments.dry_run:
        output_directory = (
            ROOT
            / "data"
            / "interim"
            / "iscxvpn2016"
            / "flow_dryrun"
        )

        stats_directory = (
            ARTIFACTS
            / "iscx_flow_dryrun_stats"
        )

        log_directory = (
            ARTIFACTS
            / "iscx_flow_dryrun_logs"
        )

        temp_directory = (
            ROOT
            / "data"
            / "interim"
            / "iscxvpn2016"
            / "temp_dryrun"
        )

        artifact_prefix = (
            "iscx_flow_dryrun"
        )

    else:
        output_directory = (
            ROOT
            / "data"
            / "processed"
            / "iscxvpn2016"
            / "flows"
        )

        stats_directory = (
            ARTIFACTS
            / "iscx_flow_stats"
        )

        log_directory = (
            ARTIFACTS
            / "iscx_flow_logs"
        )

        temp_directory = (
            ROOT
            / "data"
            / "interim"
            / "iscxvpn2016"
            / "temp_capture"
        )

        artifact_prefix = (
            "iscx_flow"
        )

    if not arguments.resume:
        for directory in [
            output_directory,
            stats_directory,
            log_directory
        ]:
            if directory.exists():
                shutil.rmtree(
                    directory
                )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    stats_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    processed = 0
    skipped_existing = 0

    for index, row in enumerate(
        rows,
        start=1
    ):
        base = output_basename(
            row
        )

        output_path = (
            output_directory
            / f"{base}.parquet"
        )

        stats_path = (
            stats_directory
            / f"{base}.json"
        )

        if (
            arguments.resume
            and output_path.exists()
            and stats_path.exists()
        ):
            print(
                f"SKIP_EXISTING="
                f"{index}/{len(rows)} "
                f"{row['archive']}::"
                f"{row['capture']}"
            )

            skipped_existing += 1
            continue

        print(
            f"PROCESSING="
            f"{index}/{len(rows)} "
            f"{row['archive']}::"
            f"{row['capture']} "
            f"DOMAIN={row['domain']} "
            f"ACTIVITY={row['activity']} "
            f"FOLD={row['fold']}"
        )

        result = process_capture(
            row,
            output_directory,
            stats_directory,
            log_directory,
            temp_directory
        )

        processed += 1

        print(
            f"CAPTURE={result['capture']} "
            f"PACKETS={result['eligible_packets']} "
            f"SKIPPED={result['skipped_packets']} "
            f"FLOWS={result['flows_total']} "
            f"GE3={result['flows_ge3']} "
            f"GE5={result['flows_ge5']} "
            f"GE10={result['flows_ge10']} "
            f"GE20={result['flows_ge20']} "
            f"GE50={result['flows_ge50']}"
        )

    (
        class_path,
        fold_path,
        capture_path
    ) = generate_summaries(
        output_directory,
        stats_directory,
        artifact_prefix
    )

    protocol = {
        "dataset": "ISCXVPN2016",
        "extractor_version": 2,
        "selected_capture_count": 133,
        "selected_activities": [
            "chat",
            "file_transfer",
            "streaming",
            "voip"
        ],
        "domain": [
            "nonvpn",
            "vpn"
        ],
        "eligible_network_traffic": (
            "IPv4/IPv6 TCP and UDP"
        ),
        "bidirectional_flow_key": (
            "protocol plus canonicalized "
            "IP-port endpoint pair"
        ),
        "direction_encoding": (
            "+1 in the direction of the first "
            "observed packet of the flow, "
            "-1 in the reverse direction"
        ),
        "segment_identifier": (
            "frame number of the first packet "
            "in the idle-timeout-delimited segment"
        ),
        "packet_size": (
            "observed Ethernet frame length "
            "from frame.len"
        ),
        "interarrival_time_unit": "seconds",
        "flow_idle_timeout_seconds": (
            FLOW_IDLE_TIMEOUT_SECONDS
        ),
        "stored_sequence_packets": (
            MAX_SEQUENCE_PACKETS
        ),
        "payload_bytes_used": False,
        "fold_assignment": (
            "inherited unchanged from "
            "iscxvpn_protocol_manifest.csv"
        ),
        "raw_storage_policy": (
            "one selected ZIP member is temporarily "
            "materialized using the 7-Zip CLI, "
            "processed, then deleted"
        ),
        "archive_extraction_reason": (
            "Python 3.12 zipfile cannot decompress "
            "the compression method used by the "
            "original ISCXVPN2016 archives"
        )
    }

    protocol_path = (
        ARTIFACTS
        / f"{artifact_prefix}_protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2
        )
        + "\n"
    )

    final_parquets = len(
        list(
            output_directory.glob(
                "*.parquet"
            )
        )
    )

    stats_files = len(
        list(
            stats_directory.glob(
                "*.json"
            )
        )
    )

    print()
    print(
        f"CAPTURES_REQUESTED={len(rows)}"
    )

    print(
        f"CAPTURES_PROCESSED_NOW={processed}"
    )

    print(
        f"CAPTURES_SKIPPED_EXISTING="
        f"{skipped_existing}"
    )

    print(
        f"OUTPUT_PARQUETS={final_parquets}"
    )

    print(
        f"CAPTURE_STATS_FILES="
        f"{stats_files}"
    )

    print(
        "PAYLOAD_BYTES_USED=FALSE"
    )

    print(
        f"MAX_SEQUENCE_PACKETS="
        f"{MAX_SEQUENCE_PACKETS}"
    )

    print(
        f"FLOW_IDLE_TIMEOUT_SECONDS="
        f"{FLOW_IDLE_TIMEOUT_SECONDS}"
    )

    print(
        f"CLASS_SUMMARY={class_path}"
    )

    print(
        f"FOLD_SUMMARY={fold_path}"
    )

    print(
        f"CAPTURE_SUMMARY={capture_path}"
    )

    print(
        f"PROTOCOL={protocol_path}"
    )

    print(
        "ISCX_FLOW_EXTRACTION=PASS"
    )


if __name__ == "__main__":
    main()