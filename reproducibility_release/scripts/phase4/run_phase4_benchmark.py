import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.node import RemoteController


ROOT = Path(__file__).resolve().parents[2]

ML_PYTHON = (
    ROOT
    / ".venv-ml"
    / "bin"
    / "python"
)

SERVER_SCRIPT = (
    ROOT
    / "scripts"
    / "phase4"
    / "tcp_microflow_server.py"
)

CLIENT_SCRIPT = (
    ROOT
    / "scripts"
    / "phase4"
    / "tcp_microflow_client.py"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase4"
    / "benchmark"
)


def parse_ping(
    output
):
    loss_match = re.search(
        r"([\d.]+)% packet loss",
        output
    )

    rtt_match = re.search(
        r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms",
        output
    )

    if (
        loss_match is None
        or rtt_match is None
    ):
        raise RuntimeError(
            f"Ping parse failure:\n{output}"
        )

    return (
        float(
            loss_match.group(
                1
            )
        ),
        float(
            rtt_match.group(
                2
            )
        )
    )


def parse_iperf(
    output
):
    return_code_match = re.search(
        r"__IPERF_RC=(\d+)",
        output
    )

    rates = re.findall(
        r"([\d.]+)\s+Mbits/sec",
        output
    )

    if (
        return_code_match is None
        or not rates
    ):
        raise RuntimeError(
            f"Iperf parse failure:\n{output}"
        )

    return (
        int(
            return_code_match.group(
                1
            )
        ),
        float(
            rates[
                -1
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "forwarding",
            "features",
            "inference"
        ],
        required=True
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=5
    )

    parser.add_argument(
        "--microflows",
        type=int,
        default=500
    )

    arguments = parser.parse_args()

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        build=False,
        autoSetMacs=True
    )

    h3 = None

    result = {
        "mode": arguments.mode,
        "network_trials": [],
        "microflows": None
    }

    try:
        controller = net.addController(
            "c0",
            controller=RemoteController,
            ip="127.0.0.1",
            port=6653
        )

        h1 = net.addHost(
            "h1"
        )

        h2 = net.addHost(
            "h2"
        )

        h3 = net.addHost(
            "h3"
        )

        h4 = net.addHost(
            "h4"
        )

        s1 = net.addSwitch(
            "s1",
            dpid="0000000000000001",
            protocols="OpenFlow13"
        )

        s2 = net.addSwitch(
            "s2",
            dpid="0000000000000002",
            protocols="OpenFlow13"
        )

        net.addLink(
            h1,
            s1,
            bw=100,
            delay="1ms"
        )

        net.addLink(
            h2,
            s1,
            bw=100,
            delay="1ms"
        )

        net.addLink(
            s1,
            s2,
            bw=100,
            delay="1ms"
        )

        net.addLink(
            h3,
            s2,
            bw=100,
            delay="1ms"
        )

        net.addLink(
            h4,
            s2,
            bw=100,
            delay="1ms"
        )

        net.build()

        controller.start()

        s1.start(
            [
                controller
            ]
        )

        s2.start(
            [
                controller
            ]
        )

        time.sleep(
            2
        )

        h3.cmd(
            "pkill -x iperf >/dev/null 2>&1 || true"
        )

        h3.cmd(
            "iperf -s -p 5001 "
            "> /tmp/phase4b_iperf_server.log "
            "2>&1 &"
        )

        time.sleep(
            1
        )

        for trial in range(
            arguments.trials
        ):
            ping_output = h1.cmd(
                "ping -c 20 -i 0.05 -W 1 10.0.0.3"
            )

            (
                packet_loss,
                average_rtt
            ) = parse_ping(
                ping_output
            )

            iperf_output = h1.cmd(
                "timeout 10s "
                "iperf "
                "-c 10.0.0.3 "
                "-p 5001 "
                "-t 3 "
                "-f m "
                "; echo __IPERF_RC=$?"
            )

            (
                return_code,
                throughput
            ) = parse_iperf(
                iperf_output
            )

            row = {
                "trial": trial,
                "packet_loss_percent": packet_loss,
                "ping_average_rtt_ms": average_rtt,
                "iperf_return_code": return_code,
                "iperf_throughput_mbit_s": throughput
            }

            result[
                "network_trials"
            ].append(
                row
            )

            print(
                f"MODE={arguments.mode} "
                f"TRIAL={trial} "
                f"LOSS={packet_loss:.2f} "
                f"RTT_MS={average_rtt:.6f} "
                f"THROUGHPUT_MBIT_S={throughput:.6f}",
                flush=True
            )

        server_command = (
            f"{ML_PYTHON} "
            f"{SERVER_SCRIPT} "
            f"--connections {arguments.microflows} "
            "> /tmp/phase4b_microflow_server.log "
            "2>&1 &"
        )

        h3.cmd(
            server_command
        )

        time.sleep(
            1
        )

        client_command = (
            f"{ML_PYTHON} "
            f"{CLIENT_SCRIPT} "
            f"--host 10.0.0.3 "
            f"--connections {arguments.microflows}"
        )

        microflow_output = h1.cmd(
            client_command
        )

        lines = [
            line.strip()
            for line in microflow_output.splitlines()
            if line.strip().startswith(
                "{"
            )
        ]

        if not lines:
            raise RuntimeError(
                f"Microflow client output failure:\n"
                f"{microflow_output}"
            )

        microflows = json.loads(
            lines[
                -1
            ]
        )

        result[
            "microflows"
        ] = microflows

        server_output = h3.cmd(
            "cat /tmp/phase4b_microflow_server.log"
        )

        if (
            microflows[
                "completed"
            ]
            != arguments.microflows
            or microflows[
                "failures"
            ]
            != 0
        ):
            raise RuntimeError(
                f"Microflow failure: {microflows}"
            )

        if (
            any(
                row[
                    "packet_loss_percent"
                ]
                != 0.0
                for row in result[
                    "network_trials"
                ]
            )
            or any(
                row[
                    "iperf_return_code"
                ]
                != 0
                for row in result[
                    "network_trials"
                ]
            )
        ):
            raise RuntimeError(
                "Network trial failure"
            )

        result[
            "ping_average_rtt_ms_mean"
        ] = statistics.mean(
            row[
                "ping_average_rtt_ms"
            ]
            for row in result[
                "network_trials"
            ]
        )

        result[
            "iperf_throughput_mbit_s_mean"
        ] = statistics.mean(
            row[
                "iperf_throughput_mbit_s"
            ]
            for row in result[
                "network_trials"
            ]
        )

        output_path = (
            ARTIFACT_DIRECTORY
            / f"{arguments.mode}_benchmark.json"
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2
            )
            + "\n",
            encoding="utf-8"
        )

        print(
            f"MICROFLOWS_COMPLETED="
            f"{microflows['completed']}"
        )

        print(
            f"MICROFLOW_RTT_P50_MS="
            f"{microflows['latency_ms_p50']:.6f}"
        )

        print(
            f"MICROFLOW_RTT_P95_MS="
            f"{microflows['latency_ms_p95']:.6f}"
        )

        print(
            f"MICROFLOW_RTT_P99_MS="
            f"{microflows['latency_ms_p99']:.6f}"
        )

        print(
            f"BENCHMARK_JSON={output_path}"
        )

        print(
            "PHASE4_BENCHMARK=PASS"
        )

    finally:
        if h3 is not None:
            h3.cmd(
                "pkill -x iperf >/dev/null 2>&1 || true"
            )

            h3.cmd(
                "pkill -f tcp_microflow_server.py "
                ">/dev/null 2>&1 || true"
            )

        net.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"PHASE4_BENCHMARK_ERROR="
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(
            1
        )
