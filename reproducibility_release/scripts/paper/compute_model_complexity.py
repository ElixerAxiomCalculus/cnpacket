from pathlib import Path
import csv
import json
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "paper"
    / "experimental_completion"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

OPEN_SET_SCRIPT_DIRECTORY = (
    ROOT
    / "scripts"
    / "phase3"
)

sys.path.insert(
    0,
    str(
        OPEN_SET_SCRIPT_DIRECTORY
    )
)

from run_ustc_open_set import Encoder


OPEN_SET_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set_protocol.json"
)

SUPCON_CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "phase3"
    / "ustc"
    / "open_set"
    / "supcon_proto_run_0.pt"
)

OUTPUT_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "model_complexity.json"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "model_complexity.csv"
)


def count_state_dict_parameters(
    state_dict
):
    return int(
        sum(
            tensor.numel()
            for tensor in state_dict.values()
            if torch.is_tensor(
                tensor
            )
        )
    )


def find_first_module(
    model,
    module_type
):
    matches = [
        module
        for module in model.modules()
        if isinstance(
            module,
            module_type
        )
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Expected one {module_type.__name__}, "
            f"found {len(matches)}"
        )

    return matches[
        0
    ]


def calculate_encoder_macs(
    convolution,
    recurrent,
    padded_sequence_length,
    valid_sequence_length
):
    convolution_kernel_width = int(
        convolution.kernel_size[
            0
        ]
    )

    convolution_macs = int(
        padded_sequence_length
        * convolution.out_channels
        * convolution.in_channels
        * convolution_kernel_width
    )

    recurrent_directions = (
        2
        if recurrent.bidirectional
        else 1
    )

    recurrent_macs = int(
        valid_sequence_length
        * recurrent_directions
        * 4
        * recurrent.hidden_size
        * (
            recurrent.input_size
            + recurrent.hidden_size
        )
    )

    return {
        "convolution_macs": convolution_macs,
        "recurrent_macs": recurrent_macs,
        "encoder_macs": (
            convolution_macs
            + recurrent_macs
        )
    }


def main():
    protocol = json.loads(
        OPEN_SET_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    checkpoint = torch.load(
        SUPCON_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    hidden_size = int(
        protocol[
            "encoder"
        ][
            "bilstm_hidden_size_per_direction"
        ]
    )

    encoder = Encoder(
        hidden_size=hidden_size
    )

    encoder.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    encoder.eval()

    convolution = find_first_module(
        encoder,
        nn.Conv1d
    )

    recurrent = find_first_module(
        encoder,
        nn.LSTM
    )

    encoder_parameters = int(
        sum(
            parameter.numel()
            for parameter in encoder.parameters()
        )
    )

    checkpoint_state_parameters = (
        count_state_dict_parameters(
            checkpoint[
                "state_dict"
            ]
        )
    )

    if (
        encoder_parameters
        != checkpoint_state_parameters
    ):
        raise RuntimeError(
            "Checkpoint parameter count does not match encoder"
        )

    if encoder_parameters != 67008:
        raise RuntimeError(
            f"Expected 67008 encoder parameters, "
            f"found {encoder_parameters}"
        )

    embedding_dimension = int(
        recurrent.hidden_size
        * (
            2
            if recurrent.bidirectional
            else 1
        )
    )

    padded_sequence_length = 50

    scenarios = []

    for valid_sequence_length in [
        5,
        20,
        50
    ]:
        base = calculate_encoder_macs(
            convolution,
            recurrent,
            padded_sequence_length,
            valid_sequence_length
        )

        for classifier_name, class_count in [
            (
                "zero_shot_8_prototypes",
                8
            ),
            (
                "few_shot_9_prototypes",
                9
            )
        ]:
            classifier_macs = int(
                embedding_dimension
                * class_count
            )

            total_macs = int(
                base[
                    "encoder_macs"
                ]
                + classifier_macs
            )

            scenarios.append(
                {
                    "scenario": classifier_name,
                    "valid_sequence_packets": valid_sequence_length,
                    "padded_sequence_packets": padded_sequence_length,
                    "trainable_parameters": encoder_parameters,
                    "convolution_macs": base[
                        "convolution_macs"
                    ],
                    "bilstm_macs": base[
                        "recurrent_macs"
                    ],
                    "classifier_macs": classifier_macs,
                    "total_macs": total_macs,
                    "arithmetic_flops": int(
                        2
                        * total_macs
                    )
                }
            )

    closed_set_valid_sequence_length = 50

    closed_base = calculate_encoder_macs(
        convolution,
        recurrent,
        padded_sequence_length,
        closed_set_valid_sequence_length
    )

    closed_set_classifier_parameters = int(
        embedding_dimension
        * 10
        + 10
    )

    closed_set_parameters = int(
        encoder_parameters
        + closed_set_classifier_parameters
    )

    closed_set_classifier_macs = int(
        embedding_dimension
        * 10
    )

    closed_set_total_macs = int(
        closed_base[
            "encoder_macs"
        ]
        + closed_set_classifier_macs
    )

    scenarios.append(
        {
            "scenario": "closed_set_linear_10_class",
            "valid_sequence_packets": 50,
            "padded_sequence_packets": 50,
            "trainable_parameters": closed_set_parameters,
            "convolution_macs": closed_base[
                "convolution_macs"
            ],
            "bilstm_macs": closed_base[
                "recurrent_macs"
            ],
            "classifier_macs": closed_set_classifier_macs,
            "total_macs": closed_set_total_macs,
            "arithmetic_flops": int(
                2
                * closed_set_total_macs
            )
        }
    )

    if closed_set_parameters != 68298:
        raise RuntimeError(
            f"Expected 68298 closed-set parameters, "
            f"found {closed_set_parameters}"
        )

    checkpoint_size_bytes = int(
        SUPCON_CHECKPOINT_PATH.stat().st_size
    )

    result = {
        "counting_convention": {
            "mac_definition": (
                "one multiplication followed by one accumulation"
            ),
            "flop_definition": (
                "one MAC counted as two arithmetic FLOPs"
            ),
            "included": [
                "Conv1d multiply-accumulate operations",
                "BiLSTM input and recurrent matrix multiply-accumulate operations",
                "linear or prototype similarity multiply-accumulate operations"
            ],
            "excluded": [
                "activation functions",
                "sigmoid and tanh internal evaluation cost",
                "bias additions",
                "normalization",
                "packing and masking control operations",
                "memory movement",
                "HTTP and controller overhead"
            ]
        },
        "architecture": {
            "encoder_parameters": encoder_parameters,
            "closed_set_parameters": closed_set_parameters,
            "embedding_dimension": embedding_dimension,
            "conv_in_channels": int(
                convolution.in_channels
            ),
            "conv_out_channels": int(
                convolution.out_channels
            ),
            "conv_kernel_width": int(
                convolution.kernel_size[
                    0
                ]
            ),
            "bilstm_input_size": int(
                recurrent.input_size
            ),
            "bilstm_hidden_size_per_direction": int(
                recurrent.hidden_size
            ),
            "bilstm_bidirectional": bool(
                recurrent.bidirectional
            ),
            "supcon_checkpoint_size_bytes": checkpoint_size_bytes,
            "supcon_checkpoint_size_kib": (
                checkpoint_size_bytes
                / 1024.0
            )
        },
        "scenarios": scenarios
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(
            result,
            indent=2
        )
        + "\n",
        encoding="utf-8"
    )

    with OUTPUT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                scenarios[
                    0
                ].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            scenarios
        )

    print(
        "===== ARCHITECTURE ====="
    )

    print(
        f"ENCODER_PARAMETERS={encoder_parameters}"
    )

    print(
        f"CLOSED_SET_PARAMETERS={closed_set_parameters}"
    )

    print(
        f"EMBEDDING_DIMENSION={embedding_dimension}"
    )

    print(
        f"SUPCON_CHECKPOINT_BYTES={checkpoint_size_bytes}"
    )

    print(
        f"SUPCON_CHECKPOINT_KIB="
        f"{checkpoint_size_bytes / 1024.0:.3f}"
    )

    print()

    print(
        "===== COMPLEXITY ====="
    )

    for row in scenarios:
        print(
            f"SCENARIO={row['scenario']} "
            f"N={row['valid_sequence_packets']} "
            f"PARAMETERS={row['trainable_parameters']} "
            f"MACS={row['total_macs']} "
            f"FLOPS={row['arithmetic_flops']}"
        )

    print()

    print(
        f"OUTPUT_JSON={OUTPUT_JSON_PATH}"
    )

    print(
        f"OUTPUT_CSV={OUTPUT_CSV_PATH}"
    )

    print(
        "MODEL_COMPLEXITY_AUDIT=PASS"
    )


if __name__ == "__main__":
    main()
