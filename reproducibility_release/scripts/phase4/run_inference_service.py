import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

PHASE3_SCRIPT_DIRECTORY = (
    ROOT
    / "scripts"
    / "phase3"
)

sys.path.insert(
    0,
    str(
        PHASE3_SCRIPT_DIRECTORY
    )
)

from run_ustc_open_set import Encoder


MODEL_PATH = (
    ROOT
    / "models"
    / "phase3"
    / "ustc"
    / "open_set"
    / "supcon_proto_run_0.pt"
)

REPORT_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set"
    / "supcon_proto_run_0_report.json"
)

PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set_protocol.json"
)

FROZEN_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set"
    / "open_set_sha256.txt"
)


def sha256_file(
    path
):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def expected_hash(
    target
):
    relative = str(
        target.relative_to(
            ROOT
        )
    )

    for line in FROZEN_MANIFEST_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        parts = line.split(
            None,
            1
        )

        if len(
            parts
        ) != 2:
            continue

        checksum = parts[
            0
        ]

        path = parts[
            1
        ].lstrip(
            "*"
        )

        if path == relative:
            return checksum

    raise RuntimeError(
        f"Frozen hash missing for {relative}"
    )


def verify_frozen_file(
    path
):
    expected = expected_hash(
        path
    )

    observed = sha256_file(
        path
    )

    if observed != expected:
        raise RuntimeError(
            f"Frozen file mismatch: {path}"
        )

    return observed


class Runtime:
    def __init__(
        self,
        device
    ):
        verify_frozen_file(
            MODEL_PATH
        )

        verify_frozen_file(
            REPORT_PATH
        )

        protocol = json.loads(
            PROTOCOL_PATH.read_text(
                encoding="utf-8"
            )
        )

        report = json.loads(
            REPORT_PATH.read_text(
                encoding="utf-8"
            )
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

        self.device = torch.device(
            device
        )

        self.normalization = checkpoint[
            "normalization"
        ]

        self.known_families = list(
            checkpoint[
                "known_families"
            ]
        )

        self.threshold = float(
            report[
                "threshold_calibration"
            ][
                "threshold"
            ]
        )

        self.encoder = Encoder(
            hidden_size=int(
                protocol[
                    "encoder"
                ][
                    "bilstm_hidden_size_per_direction"
                ]
            )
        )

        self.encoder.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

        self.encoder = self.encoder.to(
            self.device
        )

        self.encoder.eval()

        prototypes = np.asarray(
            checkpoint[
                "known_prototypes"
            ],
            dtype=np.float32
        )

        self.prototypes = torch.from_numpy(
            prototypes
        ).to(
            self.device
        )

        self.prototypes = F.normalize(
            self.prototypes,
            p=2,
            dim=1
        )

        self.model_hash = sha256_file(
            MODEL_PATH
        )

        self._warmup()

    def _encode(
        self,
        sizes,
        iats
    ):
        length = len(
            sizes
        )

        if (
            length < 3
            or length > 50
        ):
            raise ValueError(
                f"sequence length must be 3..50, got {length}"
            )

        if len(
            iats
        ) != length:
            raise ValueError(
                "size/IAT length mismatch"
            )

        values = np.zeros(
            (
                1,
                50,
                2
            ),
            dtype=np.float32
        )

        size_values = np.asarray(
            sizes,
            dtype=np.float32
        )

        iat_values = np.asarray(
            iats,
            dtype=np.float64
        )

        if not np.isfinite(
            size_values
        ).all():
            raise ValueError(
                "non-finite packet sizes"
            )

        if not np.isfinite(
            iat_values
        ).all():
            raise ValueError(
                "non-finite IAT values"
            )

        if (
            iat_values
            < 0
        ).any():
            raise ValueError(
                "negative IAT value"
            )

        values[
            0,
            :length,
            0
        ] = (
            size_values
            / np.float32(
                self.normalization[
                    "signed_size_rms"
                ]
            )
        )

        values[
            0,
            :length,
            1
        ] = (
            (
                np.log1p(
                    iat_values
                )
                - self.normalization[
                    "iat_log1p_mean"
                ]
            )
            / self.normalization[
                "iat_log1p_std"
            ]
        ).astype(
            np.float32
        )

        inputs = torch.from_numpy(
            values
        ).to(
            self.device
        )

        lengths = torch.tensor(
            [
                length
            ],
            dtype=torch.int64
        )

        return (
            inputs,
            lengths
        )

    def infer(
        self,
        sizes,
        iats
    ):
        service_start = time.perf_counter_ns()

        (
            inputs,
            lengths
        ) = self._encode(
            sizes,
            iats
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        model_start = time.perf_counter_ns()

        with torch.inference_mode():
            embedding = self.encoder(
                inputs,
                lengths
            )

            embedding = F.normalize(
                embedding,
                p=2,
                dim=1
            )

            similarities = torch.clamp(
                embedding
                @ self.prototypes.T,
                min=-1.0,
                max=1.0
            )

            maximum_similarity, index = (
                similarities.max(
                    dim=1
                )
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        model_end = time.perf_counter_ns()

        maximum_similarity = float(
            maximum_similarity.item()
        )

        index = int(
            index.item()
        )

        unknown_score = (
            1.0
            - maximum_similarity
        )

        predicted_known_family = (
            self.known_families[
                index
            ]
        )

        prediction = (
            "unknown"
            if unknown_score
            > self.threshold
            else predicted_known_family
        )

        service_end = time.perf_counter_ns()

        return {
            "prediction": prediction,
            "nearest_known_family": predicted_known_family,
            "maximum_cosine_similarity": maximum_similarity,
            "unknown_score": unknown_score,
            "threshold": self.threshold,
            "model_us": (
                model_end
                - model_start
            )
            / 1000.0,
            "service_us": (
                service_end
                - service_start
            )
            / 1000.0
        }

    def _warmup(
        self
    ):
        for _ in range(
            20
        ):
            self.infer(
                [
                    64,
                    -64,
                    128,
                    -128,
                    256
                ],
                [
                    0.0,
                    0.001,
                    0.001,
                    0.001,
                    0.001
                ]
            )


class Handler(
    BaseHTTPRequestHandler
):
    runtime = None

    def _write_json(
        self,
        status,
        payload
    ):
        body = json.dumps(
            payload,
            separators=(
                ",",
                ":"
            )
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(
                len(
                    body
                )
            )
        )

        self.send_header(
            "Connection",
            "close"
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    def do_GET(
        self
    ):
        if self.path != "/health":
            self._write_json(
                404,
                {
                    "error": "not_found"
                }
            )
            return

        self._write_json(
            200,
            {
                "status": "ready",
                "device": str(
                    self.runtime.device
                ),
                "model_sha256": self.runtime.model_hash,
                "known_families": self.runtime.known_families,
                "threshold": self.runtime.threshold
            }
        )

    def do_POST(
        self
    ):
        if self.path != "/infer":
            self._write_json(
                404,
                {
                    "error": "not_found"
                }
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            body = self.rfile.read(
                content_length
            )

            request = json.loads(
                body.decode(
                    "utf-8"
                )
            )

            result = self.runtime.infer(
                request[
                    "signed_packet_sizes"
                ],
                request[
                    "interarrival_seconds"
                ]
            )

            self._write_json(
                200,
                result
            )

        except Exception as exc:
            self._write_json(
                400,
                {
                    "error": type(
                        exc
                    ).__name__,
                    "message": str(
                        exc
                    )
                }
            )

    def log_message(
        self,
        format,
        *args
    ):
        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765
    )

    parser.add_argument(
        "--device",
        default="cuda:0"
    )

    arguments = parser.parse_args()

    if arguments.device.startswith(
        "cuda"
    ) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    runtime = Runtime(
        arguments.device
    )

    Handler.runtime = runtime

    server = HTTPServer(
        (
            arguments.host,
            arguments.port
        ),
        Handler
    )

    print(
        f"PHASE4_INFERENCE_SERVICE=READY "
        f"HOST={arguments.host} "
        f"PORT={arguments.port} "
        f"DEVICE={arguments.device} "
        f"MODEL_SHA256={runtime.model_hash}",
        flush=True
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
