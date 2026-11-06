import argparse
import json
import socket
import time

import numpy as np


def percentile(
    values,
    quantile
):
    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64
            ),
            quantile
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        required=True
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5002
    )

    parser.add_argument(
        "--connections",
        type=int,
        required=True
    )

    parser.add_argument(
        "--request-bytes",
        type=int,
        default=256
    )

    parser.add_argument(
        "--response-bytes",
        type=int,
        default=64
    )

    arguments = parser.parse_args()

    request = b"Q" * arguments.request_bytes

    latencies_ms = []
    failures = []

    for index in range(
        arguments.connections
    ):
        start_ns = time.perf_counter_ns()

        try:
            with socket.create_connection(
                (
                    arguments.host,
                    arguments.port
                ),
                timeout=2.0
            ) as connection:
                connection.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1
                )

                connection.sendall(
                    request
                )

                received = 0

                while received < arguments.response_bytes:
                    block = connection.recv(
                        arguments.response_bytes
                        - received
                    )

                    if not block:
                        raise RuntimeError(
                            "short response"
                        )

                    received += len(
                        block
                    )

            end_ns = time.perf_counter_ns()

            latencies_ms.append(
                (
                    end_ns
                    - start_ns
                )
                / 1_000_000.0
            )

        except Exception as exc:
            failures.append(
                {
                    "index": index,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                }
            )

    output = {
        "requested": arguments.connections,
        "completed": len(
            latencies_ms
        ),
        "failures": len(
            failures
        ),
        "latency_ms_mean": float(
            np.mean(
                latencies_ms
            )
        ),
        "latency_ms_std": float(
            np.std(
                latencies_ms,
                ddof=1
            )
        ),
        "latency_ms_p50": percentile(
            latencies_ms,
            50
        ),
        "latency_ms_p95": percentile(
            latencies_ms,
            95
        ),
        "latency_ms_p99": percentile(
            latencies_ms,
            99
        )
    }

    print(
        json.dumps(
            output,
            separators=(
                ",",
                ":"
            )
        )
    )

    if failures:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
