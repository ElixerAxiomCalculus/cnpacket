import argparse
import socket


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="0.0.0.0"
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

    response = b"R" * arguments.response_bytes

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server.bind(
            (
                arguments.host,
                arguments.port
            )
        )

        server.listen(
            128
        )

        completed = 0

        while completed < arguments.connections:
            connection, _ = server.accept()

            with connection:
                connection.settimeout(
                    2.0
                )

                received = 0

                while received < arguments.request_bytes:
                    block = connection.recv(
                        arguments.request_bytes
                        - received
                    )

                    if not block:
                        break

                    received += len(
                        block
                    )

                if received != arguments.request_bytes:
                    continue

                connection.sendall(
                    response
                )

                completed += 1

    print(
        f"SERVER_COMPLETED={completed}"
    )


if __name__ == "__main__":
    main()
