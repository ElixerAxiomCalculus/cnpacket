import argparse
import subprocess
import time

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.node import RemoteController


def controller_connected(switch):
    value = switch.cmd(
        f"ovs-vsctl get Controller {switch.name} is_connected"
    )
    return value.strip().lower() == "true"


def wait_for_controller(switches, timeout_seconds=10):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if all(controller_connected(switch) for switch in switches):
            return True

        time.sleep(0.5)

    return False


def count_flows(flow_dump):
    return sum(
        1
        for line in flow_dump.splitlines()
        if "actions=" in line
    )


def run_bounded_iperf(client, server, seconds=3):
    server_process = server.popen(
        [
            "iperf",
            "-s",
            "-p",
            "5001"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    time.sleep(1)

    try:
        client_process = client.popen(
            [
                "timeout",
                "10",
                "iperf",
                "-c",
                server.IP(),
                "-p",
                "5001",
                "-t",
                str(seconds),
                "-f",
                "m"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        try:
            client_output, _ = client_process.communicate(
                timeout=12
            )
        except subprocess.TimeoutExpired:
            client_process.kill()
            client_output, _ = client_process.communicate()
            return 124, client_output, ""

        return_code = client_process.returncode

    finally:
        server_process.terminate()

        try:
            server_output, _ = server_process.communicate(
                timeout=2
            )
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_output, _ = server_process.communicate()

    return return_code, client_output, server_output


def build_network():
    network = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=False,
        build=False
    )

    controller = network.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653
    )

    s1 = network.addSwitch("s1", protocols="OpenFlow13")
    s2 = network.addSwitch("s2", protocols="OpenFlow13")
    h1 = network.addHost("h1", ip="10.0.0.1/24")
    h2 = network.addHost("h2", ip="10.0.0.2/24")
    h3 = network.addHost("h3", ip="10.0.0.3/24")
    h4 = network.addHost("h4", ip="10.0.0.4/24")

    network.addLink(h1, s1, cls=TCLink, bw=100, delay="1ms")
    network.addLink(h2, s1, cls=TCLink, bw=100, delay="1ms")
    network.addLink(s1, s2, cls=TCLink, bw=100, delay="1ms")
    network.addLink(h3, s2, cls=TCLink, bw=100, delay="1ms")
    network.addLink(h4, s2, cls=TCLink, bw=100, delay="1ms")

    network.build()
    controller.start()
    s1.start([controller])
    s2.start([controller])

    return network, s1, s2, h1, h4


def run_smoke_test():
    network, s1, s2, h1, h4 = build_network()

    try:
        switches = [s1, s2]
        connected = wait_for_controller(switches, timeout_seconds=10)
        s1_connected = controller_connected(s1)
        s2_connected = controller_connected(s2)

        print(f"CONTROLLER_CONNECTED={str(connected).upper()}")
        print(f"S1_CONTROLLER_CONNECTED={str(s1_connected).upper()}")
        print(f"S2_CONTROLLER_CONNECTED={str(s2_connected).upper()}")

        loss = network.pingAll()
        print(f"PING_LOSS_PERCENT={loss:.2f}")

        print(
            "IPERF_TEST_BEGIN h1=10.0.0.1 h4=10.0.0.4 "
            "duration_seconds=3 timeout_seconds=10"
        )

        (
            iperf_return_code,
            iperf_client_output,
            iperf_server_output
        ) = run_bounded_iperf(h1, h4, seconds=3)

        print(f"IPERF_RETURN_CODE={iperf_return_code}")
        print("IPERF_CLIENT_OUTPUT_BEGIN")
        print(iperf_client_output.strip())
        print("IPERF_CLIENT_OUTPUT_END")
        print("IPERF_SERVER_OUTPUT_BEGIN")
        print(iperf_server_output.strip())
        print("IPERF_SERVER_OUTPUT_END")

        iperf_passed = (
            iperf_return_code == 0
            and "bits/sec" in iperf_client_output
        )
        print(f"IPERF_TEST_PASS={str(iperf_passed).upper()}")

        s1_flows = s1.cmd("ovs-ofctl -O OpenFlow13 dump-flows s1")
        s2_flows = s2.cmd("ovs-ofctl -O OpenFlow13 dump-flows s2")

        print("S1_FLOW_DUMP_BEGIN")
        print(s1_flows.strip())
        print("S1_FLOW_DUMP_END")
        print("S2_FLOW_DUMP_BEGIN")
        print(s2_flows.strip())
        print("S2_FLOW_DUMP_END")

        s1_flow_count = count_flows(s1_flows)
        s2_flow_count = count_flows(s2_flows)
        print(f"S1_FLOW_COUNT={s1_flow_count}")
        print(f"S2_FLOW_COUNT={s2_flow_count}")

        passed = (
            connected
            and s1_connected
            and s2_connected
            and loss == 0.0
            and iperf_passed
            and s1_flow_count >= 2
            and s2_flow_count >= 2
        )

        if passed:
            print("PHASE1_SMOKE_TEST=PASS")
            return 0

        print("PHASE1_SMOKE_TEST=FAIL")
        return 1

    finally:
        network.stop()


def run_interactive():
    network, _, _, _, _ = build_network()

    try:
        CLI(network)
    finally:
        network.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "interactive"],
        default="smoke"
    )
    arguments = parser.parse_args()
    setLogLevel("info")

    if arguments.mode == "interactive":
        run_interactive()
        return

    raise SystemExit(run_smoke_test())


if __name__ == "__main__":
    main()
