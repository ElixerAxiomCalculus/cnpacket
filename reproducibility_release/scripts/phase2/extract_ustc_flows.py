import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "phase2" / "ustc_capture_manifest.csv"

FLOW_IDLE_TIMEOUT_SECONDS = 120.0
MAX_SEQUENCE_PACKETS = 50
MIN_PACKETS_CANDIDATE = 3
WRITE_BATCH_SIZE = 10_000

SCHEMA = pa.schema(
    [
        ("traffic_type", pa.string()),
        ("class_name", pa.string()),
        ("capture", pa.string()),
        ("flow_id", pa.string()),
        ("protocol", pa.string()),
        ("endpoint_a_ip", pa.string()),
        ("endpoint_a_port", pa.int32()),
        ("endpoint_b_ip", pa.string()),
        ("endpoint_b_port", pa.int32()),
        ("segment", pa.int32()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration_seconds", pa.float64()),
        ("packet_count", pa.int64()),
        ("byte_count", pa.int64()),
        ("sequence_packets", pa.int32()),
        (
            "signed_packet_sizes",
            pa.list_(pa.int32())
        ),
        (
            "interarrival_seconds",
            pa.list_(pa.float64())
        ),
        ("valid_ge3", pa.bool_())
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
    signed_packet_sizes: list = field(
        default_factory=list
    )
    interarrival_seconds: list = field(
        default_factory=list
    )


class ParquetSink:
    def __init__(self, path):
        self.path = path
        self.writer = None
        self.buffer = []
        self.rows_written = 0

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

        self.rows_written += len(
            self.buffer
        )

        self.buffer.clear()

    def close(self):
        self.flush()

        if self.writer is not None:
            self.writer.close()
        else:
            table = pa.Table.from_pylist(
                [],
                schema=SCHEMA
            )

            pq.write_table(
                table,
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


def stable_flow_id(
    capture,
    state
):
    text = (
        f"{capture}|"
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


def state_to_row(
    state,
    traffic_type,
    class_name,
    capture
):
    duration = max(
        0.0,
        state.last_time
        - state.start_time
    )

    sequence_packets = len(
        state.signed_packet_sizes
    )

    return {
        "traffic_type": traffic_type,
        "class_name": class_name,
        "capture": capture,
        "flow_id": stable_flow_id(
            capture,
            state
        ),
        "protocol": state.protocol,
        "endpoint_a_ip": (
            state.endpoint_a_ip
        ),
        "endpoint_a_port": (
            state.endpoint_a_port
        ),
        "endpoint_b_ip": (
            state.endpoint_b_ip
        ),
        "endpoint_b_port": (
            state.endpoint_b_port
        ),
        "segment": state.segment,
        "start_time": state.start_time,
        "end_time": state.last_time,
        "duration_seconds": duration,
        "packet_count": state.packet_count,
        "byte_count": state.byte_count,
        "sequence_packets": (
            sequence_packets
        ),
        "signed_packet_sizes": (
            state.signed_packet_sizes
        ),
        "interarrival_seconds": (
            state.interarrival_seconds
        ),
        "valid_ge3": (
            state.packet_count
            >= MIN_PACKETS_CANDIDATE
        )
    }


def process_capture(
    manifest_row,
    output_directory,
    log_directory
):
    capture_path = (
        ROOT
        / manifest_row["capture"]
    )

    if not capture_path.exists():
        raise RuntimeError(
            f"Missing capture: {capture_path}"
        )

    traffic_type = (
        manifest_row["traffic_type"]
    )

    class_name = (
        manifest_row["class_name"]
    )

    capture_name = (
        manifest_row["filename"]
    )

    output_name = (
        f"{sanitize(traffic_type)}__"
        f"{sanitize(class_name)}__"
        f"{sanitize(Path(capture_name).stem)}"
        ".parquet"
    )

    output_path = (
        output_directory
        / output_name
    )

    stderr_path = (
        log_directory
        / f"{output_name}.tshark.log"
    )

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
    valid_ge3 = 0

    sink = ParquetSink(
        output_path
    )

    def finalize(state):
        nonlocal total_flows
        nonlocal valid_ge3

        row = state_to_row(
            state,
            traffic_type,
            class_name,
            capture_name
        )

        sink.append(row)

        total_flows += 1

        if row["valid_ge3"]:
            valid_ge3 += 1

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
                "Unable to read tshark stdout"
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
                    iter(active.items())
                )

                if (
                    timestamp
                    - oldest_state.last_time
                    <= FLOW_IDLE_TIMEOUT_SECONDS
                ):
                    break

                finalize(oldest_state)
                del active[oldest_key]   

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
                endpoint_a = (
                    source_ip,
                    source_port
                )
                endpoint_b = (
                    destination_ip,
                    destination_port
                )
                state = FlowState(
                    protocol=protocol,
                    endpoint_a_ip=source_ip,
                    endpoint_a_port=source_port,
                    endpoint_b_ip=destination_ip,
                    endpoint_b_port=destination_port,
                    segment=int(frame_number),
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
                interarrival = max(
                    0.0,
                    timestamp
                    - state.last_time
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
                    and destination_endpoint == endpoint_b
                ):
                    direction = 1
                elif (
                    source_endpoint == endpoint_b
                    and destination_endpoint == endpoint_a
                ):
                    direction = -1
                else:
                    raise RuntimeError(
                        "Canonical flow-key inconsistency"
                    )

            state.packet_count += 1
            state.byte_count += frame_length

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
            f"{capture_name} "
            f"with code {return_code}\n"
            f"{message[-4000:]}"
        )

    for state in active.values():
        finalize(state)

    sink.close()

    return {
        "traffic_type": traffic_type,
        "class_name": class_name,
        "capture": capture_name,
        "eligible_packets": (
            eligible_packets
        ),
        "skipped_packets": (
            skipped_packets
        ),
        "flows_total": total_flows,
        "flows_ge3": valid_ge3,
        "output_parquet": str(
            output_path.relative_to(ROOT)
        )
    }


def generate_summary(
    output_directory,
    artifact_prefix
):
    parquet_glob = str(
        output_directory / "*.parquet"
    ).replace(
        "'",
        "''"
    )

    connection = duckdb.connect()

    summary = connection.execute(
        f"""
        SELECT
            traffic_type,
            class_name,
            COUNT(*) AS flows_total,
            SUM(
                CASE
                    WHEN packet_count >= 3
                    THEN 1
                    ELSE 0
                END
            ) AS flows_ge3,
            SUM(
                CASE
                    WHEN packet_count >= 5
                    THEN 1
                    ELSE 0
                END
            ) AS flows_ge5,
            SUM(
                CASE
                    WHEN packet_count >= 10
                    THEN 1
                    ELSE 0
                END
            ) AS flows_ge10,
            SUM(
                CASE
                    WHEN packet_count >= 20
                    THEN 1
                    ELSE 0
                END
            ) AS flows_ge20,
            SUM(
                CASE
                    WHEN packet_count >= 50
                    THEN 1
                    ELSE 0
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
            traffic_type,
            class_name
        ORDER BY
            traffic_type,
            class_name
        """
    ).df()

    summary_path = (
        ROOT
        / "artifacts"
        / "phase2"
        / f"{artifact_prefix}_class_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    connection.close()

    return summary_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only-class",
        default=None
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    arguments = parser.parse_args()

    with MANIFEST.open(
        newline="",
        encoding="utf-8"
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if arguments.only_class:
        rows = [
            row
            for row in rows
            if row["class_name"]
            == arguments.only_class
        ]

    if not rows:
        raise RuntimeError(
            "No captures matched selection"
        )

    if arguments.dry_run:
        output_directory = (
            ROOT
            / "data"
            / "interim"
            / "ustc-tfc2016"
            / "flow_dryrun"
        )

        log_directory = (
            ROOT
            / "artifacts"
            / "phase2"
            / "ustc_flow_dryrun_logs"
        )

        artifact_prefix = (
            "ustc_flow_dryrun"
        )

    else:
        output_directory = (
            ROOT
            / "data"
            / "processed"
            / "ustc-tfc2016"
            / "flows"
        )

        log_directory = (
            ROOT
            / "artifacts"
            / "phase2"
            / "ustc_flow_logs"
        )

        artifact_prefix = (
            "ustc_flow"
        )

    if output_directory.exists():
        shutil.rmtree(
            output_directory
        )

    if log_directory.exists():
        shutil.rmtree(
            log_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for index, row in enumerate(
        rows,
        start=1
    ):
        print(
            f"PROCESSING={index}/{len(rows)} "
            f"{row['traffic_type']}/"
            f"{row['class_name']}/"
            f"{row['filename']}"
        )

        result = process_capture(
            row,
            output_directory,
            log_directory
        )

        results.append(
            result
        )

        print(
            f"CAPTURE={result['capture']} "
            f"PACKETS={result['eligible_packets']} "
            f"SKIPPED={result['skipped_packets']} "
            f"FLOWS={result['flows_total']} "
            f"FLOWS_GE3={result['flows_ge3']}"
        )

    capture_summary_path = (
        ROOT
        / "artifacts"
        / "phase2"
        / f"{artifact_prefix}_capture_summary.csv"
    )

    with capture_summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "traffic_type",
                "class_name",
                "capture",
                "eligible_packets",
                "skipped_packets",
                "flows_total",
                "flows_ge3",
                "output_parquet"
            ]
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    class_summary_path = generate_summary(
        output_directory,
        artifact_prefix
    )

    protocol = {
        "dataset": "USTC-TFC2016",
        "extractor_version": 2,
        "eligible_network_traffic": (
            "IPv4/IPv6 TCP and UDP"
        ),
        "bidirectional_flow_key": (
            "protocol plus canonicalized "
            "source/destination IP-port endpoints"
        ),
        "direction_encoding": (
            "+1 in the direction of the first observed "
            "packet of the flow segment, -1 in the reverse "
            "direction"
        ),
        "segment_identifier": (
            "frame number of the first packet in the "
            "idle-timeout-delimited flow segment"
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
        "candidate_minimum_packets": (
            MIN_PACKETS_CANDIDATE
        ),
        "payload_bytes_used": False,
        "split_policy": (
            "not assigned during extraction"
        )
    }

    protocol_path = (
        ROOT
        / "artifacts"
        / "phase2"
        / f"{artifact_prefix}_protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2
        )
        + "\n"
    )

    total_packets = sum(
        row["eligible_packets"]
        for row in results
    )

    total_flows = sum(
        row["flows_total"]
        for row in results
    )

    total_valid = sum(
        row["flows_ge3"]
        for row in results
    )

    print()
    print(
        f"CAPTURES_PROCESSED={len(results)}"
    )

    print(
        f"ELIGIBLE_PACKETS={total_packets}"
    )

    print(
        f"FLOWS_TOTAL={total_flows}"
    )

    print(
        f"FLOWS_GE3={total_valid}"
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
        f"CAPTURE_SUMMARY="
        f"{capture_summary_path}"
    )

    print(
        f"CLASS_SUMMARY="
        f"{class_summary_path}"
    )

    print(
        "USTC_FLOW_EXTRACTION=PASS"
    )


if __name__ == "__main__":
    main()