import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8"
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
    / "ustc_closed_model_ready.parquet"
)

PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set_protocol.json"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
)

MODEL_DIRECTORY = (
    ROOT
    / "models"
    / "phase3"
    / "ustc"
    / "closed_set"
)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size,
        dropout
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            2,
            64,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def encode(
        self,
        inputs,
        lengths
    ):
        features = self.activation(
            self.conv(
                inputs.transpose(
                    1,
                    2
                )
            )
        ).transpose(
            1,
            2
        )

        positions = torch.arange(
            features.shape[1],
            device=features.device
        ).unsqueeze(
            0
        )

        mask = (
            positions
            < lengths.to(
                features.device
            ).unsqueeze(
                1
            )
        ).unsqueeze(
            2
        )

        features = features * mask

        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (
            hidden,
            _
        ) = self.lstm(
            packed
        )

        embedding = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        return embedding

    def forward(
        self,
        inputs,
        lengths
    ):
        embedding = self.encode(
            inputs,
            lengths
        )

        return self.classifier(
            self.dropout(
                embedding
            )
        )


def set_seed(
    seed
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(
        True,
        warn_only=True
    )


def fit_normalization(
    frame
):
    size_squared_sum = 0.0
    iat_sum = 0.0
    iat_squared_sum = 0.0
    token_count = 0

    for (
        sizes,
        iats,
        length
    ) in zip(
        frame[
            "signed_packet_sizes"
        ],
        frame[
            "interarrival_seconds"
        ],
        frame[
            "sequence_packets"
        ]
    ):
        length = int(
            length
        )

        sizes = np.asarray(
            sizes,
            dtype=np.float64
        )[
            :length
        ]

        iats = np.asarray(
            iats,
            dtype=np.float64
        )[
            :length
        ]

        transformed_iats = np.log1p(
            iats
        )

        size_squared_sum += float(
            np.square(
                sizes
            ).sum()
        )

        iat_sum += float(
            transformed_iats.sum()
        )

        iat_squared_sum += float(
            np.square(
                transformed_iats
            ).sum()
        )

        token_count += length

    size_rms = math.sqrt(
        size_squared_sum
        / token_count
    )

    iat_mean = (
        iat_sum
        / token_count
    )

    iat_variance = (
        iat_squared_sum
        / token_count
        - iat_mean**2
    )

    iat_std = math.sqrt(
        max(
            iat_variance,
            0.0
        )
    )

    if (
        not np.isfinite(
            size_rms
        )
        or size_rms <= 0
    ):
        raise RuntimeError(
            "Invalid packet-size normalization"
        )

    if (
        not np.isfinite(
            iat_std
        )
        or iat_std <= 0
    ):
        raise RuntimeError(
            "Invalid IAT normalization"
        )

    return {
        "valid_training_tokens": int(
            token_count
        ),
        "signed_size_rms": float(
            size_rms
        ),
        "iat_log1p_mean": float(
            iat_mean
        ),
        "iat_log1p_std": float(
            iat_std
        )
    }


def encode_frame(
    frame,
    normalization,
    label_mapping
):
    maximum_length = 50

    inputs = np.zeros(
        (
            len(
                frame
            ),
            maximum_length,
            2
        ),
        dtype=np.float32
    )

    lengths = frame[
        "sequence_packets"
    ].to_numpy(
        dtype=np.int64,
        copy=True
    )

    labels = (
        frame[
            "class_name"
        ]
        .map(
            label_mapping
        )
        .to_numpy(
            dtype=np.int64,
            copy=True
        )
    )

    for index, (
        sizes,
        iats,
        length
    ) in enumerate(
        zip(
            frame[
                "signed_packet_sizes"
            ],
            frame[
                "interarrival_seconds"
            ],
            lengths
        )
    ):
        length = int(
            length
        )

        if (
            length < 3
            or length > maximum_length
        ):
            raise RuntimeError(
                f"Invalid sequence length {length}"
            )

        sizes = np.asarray(
            sizes,
            dtype=np.float32
        )[
            :length
        ]

        iats = np.asarray(
            iats,
            dtype=np.float64
        )[
            :length
        ]

        inputs[
            index,
            :length,
            0
        ] = (
            sizes
            / np.float32(
                normalization[
                    "signed_size_rms"
                ]
            )
        )

        inputs[
            index,
            :length,
            1
        ] = (
            (
                np.log1p(
                    iats
                )
                - normalization[
                    "iat_log1p_mean"
                ]
            )
            / normalization[
                "iat_log1p_std"
            ]
        ).astype(
            np.float32
        )

    if not np.isfinite(
        inputs
    ).all():
        raise RuntimeError(
            "Non-finite transformed inputs"
        )

    return (
        inputs,
        lengths,
        labels
    )


def make_loader(
    inputs,
    lengths,
    labels,
    batch_size,
    shuffle,
    seed
):
    dataset = TensorDataset(
        torch.from_numpy(
            inputs
        ),
        torch.from_numpy(
            lengths
        ),
        torch.from_numpy(
            labels
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator
    )


def class_weights(
    labels,
    num_classes
):
    counts = np.bincount(
        labels,
        minlength=num_classes
    )

    if (
        counts
        == 0
    ).any():
        raise RuntimeError(
            "Training split is missing a family"
        )

    weights = (
        len(
            labels
        )
        / (
            num_classes
            * counts
        )
    )

    return (
        counts,
        weights.astype(
            np.float32
        )
    )


def calculate_metrics(
    labels,
    probabilities,
    class_names
):
    num_classes = len(
        class_names
    )

    predictions = probabilities.argmax(
        axis=1
    )

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(
            range(
                num_classes
            )
        ),
        zero_division=0
    )

    result = {
        "samples": int(
            len(
                labels
            )
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(
                range(
                    num_classes
                )
            )
        ).astype(
            int
        ).tolist(),
        "per_class": {}
    }

    for (
        index,
        name
    ) in enumerate(
        class_names
    ):
        result[
            "per_class"
        ][
            name
        ] = {
            "precision": float(
                precision[
                    index
                ]
            ),
            "recall": float(
                recall[
                    index
                ]
            ),
            "f1": float(
                f1[
                    index
                ]
            ),
            "support": int(
                support[
                    index
                ]
            )
        }

    binary = label_binarize(
        labels,
        classes=list(
            range(
                num_classes
            )
        )
    )

    result[
        "roc_auc_ovr_macro"
    ] = float(
        roc_auc_score(
            labels,
            probabilities,
            labels=list(
                range(
                    num_classes
                )
            ),
            multi_class="ovr",
            average="macro"
        )
    )

    result[
        "average_precision_macro"
    ] = float(
        average_precision_score(
            binary,
            probabilities,
            average="macro"
        )
    )

    return (
        result,
        predictions
    )


def predict(
    model,
    inputs,
    lengths,
    batch_size,
    device
):
    dummy_labels = np.zeros(
        len(
            inputs
        ),
        dtype=np.int64
    )

    loader = make_loader(
        inputs,
        lengths,
        dummy_labels,
        batch_size,
        False,
        0
    )

    probability_batches = []

    model.eval()

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        for (
            batch_inputs,
            batch_lengths,
            _
        ) in loader:
            batch_inputs = batch_inputs.to(
                device,
                non_blocking=True
            )

            logits = model(
                batch_inputs,
                batch_lengths
            )

            probability_batches.append(
                torch.softmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.concatenate(
            probability_batches,
            axis=0
        ),
        elapsed
    )


def train_model(
    model,
    train_inputs,
    train_lengths,
    train_labels,
    validation_inputs,
    validation_lengths,
    validation_labels,
    weights,
    protocol,
    seed,
    device,
    class_names
):
    config = protocol[
        "training"
    ]

    batch_size = int(
        config[
            "batch_size"
        ]
    )

    loader = make_loader(
        train_inputs,
        train_lengths,
        train_labels,
        batch_size,
        True,
        seed
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            config[
                "weight_decay"
            ]
        )
    )

    best_score = float(
        "-inf"
    )

    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    torch.cuda.synchronize()

    training_start = time.perf_counter()

    for epoch in range(
        1,
        int(
            config[
                "max_epochs"
            ]
        )
        + 1
    ):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for (
            batch_inputs,
            batch_lengths,
            batch_labels
        ) in loader:
            batch_inputs = batch_inputs.to(
                device,
                non_blocking=True
            )

            batch_labels = batch_labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_inputs,
                batch_lengths
            )

            loss = criterion(
                logits,
                batch_labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    config[
                        "gradient_clip_norm"
                    ]
                )
            )

            optimizer.step()

            count = len(
                batch_labels
            )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * count
            )

            total_samples += count

        train_loss = (
            total_loss
            / total_samples
        )

        validation_probabilities, _ = predict(
            model,
            validation_inputs,
            validation_lengths,
            batch_size,
            device
        )

        validation_metrics, _ = calculate_metrics(
            validation_labels,
            validation_probabilities,
            class_names
        )

        score = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        if score > (
            best_score
            + float(
                config[
                    "early_stopping_min_delta"
                ]
            )
        ):
            best_score = score

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for (
                    key,
                    value
                ) in model.state_dict().items()
            }

            stale_epochs = 0

        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": validation_metrics[
                    "accuracy"
                ],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_macro_f1": validation_metrics[
                    "macro_f1"
                ],
                "validation_weighted_f1": validation_metrics[
                    "weighted_f1"
                ],
                "best_validation_macro_f1": best_score
            }
        )

        print(
            f"EPOCH={epoch} "
            f"TRAIN_LOSS={train_loss:.6f} "
            f"VAL_MACRO_F1={score:.6f} "
            f"BEST={best_score:.6f}"
        )

        if stale_epochs >= int(
            config[
                "early_stopping_patience"
            ]
        ):
            print(
                f"EARLY_STOPPING_EPOCH={epoch}"
            )

            break

    torch.cuda.synchronize()

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    if best_state is None:
        raise RuntimeError(
            "No validation checkpoint recorded"
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        device
    )

    return (
        model,
        history,
        best_epoch,
        best_score,
        training_seconds
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    arguments = parser.parse_args()

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    allowed_seeds = [
        int(
            seed
        )
        for seed in protocol[
            "seeds"
        ]
    ]

    if arguments.seed not in allowed_seeds:
        raise RuntimeError(
            f"Seed {arguments.seed} is outside frozen protocol"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    set_seed(
        arguments.seed
    )

    frame = pd.read_parquet(
        DATA_PATH
    )

    if len(
        frame
    ) != 229581:
        raise RuntimeError(
            f"Unexpected model-ready rows={len(frame)}"
        )

    split_counts = (
        frame[
            "closed_split"
        ]
        .value_counts()
        .to_dict()
    )

    expected_split_counts = {
        "train": 137748,
        "validation": 45915,
        "test": 45918
    }

    if split_counts != expected_split_counts:
        raise RuntimeError(
            f"Split mismatch: {split_counts}"
        )

    group_split_counts = (
        frame.groupby(
            [
                "class_name",
                "representation_fingerprint"
            ]
        )[
            "closed_split"
        ]
        .nunique()
    )

    if (
        group_split_counts
        > 1
    ).any():
        raise RuntimeError(
            "Representation leakage detected"
        )

    class_names = protocol[
        "families"
    ]

    label_mapping = {
        name: index
        for (
            index,
            name
        ) in enumerate(
            class_names
        )
    }

    train_frame = frame[
        frame[
            "closed_split"
        ]
        == "train"
    ].copy()

    validation_frame = frame[
        frame[
            "closed_split"
        ]
        == "validation"
    ].copy()

    test_frame = frame[
        frame[
            "closed_split"
        ]
        == "test"
    ].copy()

    del frame

    normalization = fit_normalization(
        train_frame
    )

    (
        train_inputs,
        train_lengths,
        train_labels
    ) = encode_frame(
        train_frame,
        normalization,
        label_mapping
    )

    (
        validation_inputs,
        validation_lengths,
        validation_labels
    ) = encode_frame(
        validation_frame,
        normalization,
        label_mapping
    )

    (
        test_inputs,
        test_lengths,
        test_labels
    ) = encode_frame(
        test_frame,
        normalization,
        label_mapping
    )

    (
        counts,
        weights
    ) = class_weights(
        train_labels,
        len(
            class_names
        )
    )

    device = torch.device(
        "cuda:0"
    )

    model = CNNBiLSTM(
        num_classes=len(
            class_names
        ),
        hidden_size=int(
            protocol[
                "model"
            ][
                "bilstm_hidden_size_per_direction"
            ]
        ),
        dropout=float(
            protocol[
                "model"
            ][
                "dropout"
            ]
        )
    ).to(
        device
    )

    parameters = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    print(
        "EXPERIMENT=ustc_closed_set"
    )

    print(
        f"SEED={arguments.seed}"
    )

    print(
        f"TRAIN_ROWS={len(train_frame)}"
    )

    print(
        f"VALIDATION_ROWS={len(validation_frame)}"
    )

    print(
        f"TEST_ROWS={len(test_frame)}"
    )

    print(
        f"TRAIN_CLASS_COUNTS={counts.tolist()}"
    )

    print(
        f"CLASS_WEIGHTS={weights.tolist()}"
    )

    print(
        f"PARAMETERS={parameters}"
    )

    print(
        f"NORMALIZATION="
        f"{json.dumps(normalization, sort_keys=True)}"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    (
        model,
        history,
        best_epoch,
        best_validation_macro_f1,
        training_seconds
    ) = train_model(
        model,
        train_inputs,
        train_lengths,
        train_labels,
        validation_inputs,
        validation_lengths,
        validation_labels,
        weights,
        protocol,
        arguments.seed,
        device,
        class_names
    )

    batch_size = int(
        protocol[
            "training"
        ][
            "batch_size"
        ]
    )

    validation_probabilities, validation_seconds = predict(
        model,
        validation_inputs,
        validation_lengths,
        batch_size,
        device
    )

    validation_metrics, _ = calculate_metrics(
        validation_labels,
        validation_probabilities,
        class_names
    )

    test_probabilities, test_seconds = predict(
        model,
        test_inputs,
        test_lengths,
        batch_size,
        device
    )

    (
        test_metrics,
        test_predictions
    ) = calculate_metrics(
        test_labels,
        test_probabilities,
        class_names
    )

    peak_gpu_memory_mib = float(
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    run_name = (
        f"cnn_bilstm_seed_{arguments.seed}"
    )

    history_path = (
        ARTIFACT_DIRECTORY
        / f"{run_name}_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    prediction_path = (
        ARTIFACT_DIRECTORY
        / f"{run_name}_test_predictions.parquet"
    )

    prediction_frame = test_frame[
        [
            "flow_id",
            "class_name",
            "capture",
            "start_time",
            "sequence_packets",
            "representation_fingerprint"
        ]
    ].reset_index(
        drop=True
    ).copy()

    prediction_frame[
        "true_label"
    ] = test_labels

    prediction_frame[
        "predicted_label"
    ] = test_predictions

    prediction_frame[
        "predicted_family"
    ] = [
        class_names[
            prediction
        ]
        for prediction in test_predictions
    ]

    for (
        index,
        class_name
    ) in enumerate(
        class_names
    ):
        prediction_frame[
            f"probability_{class_name}"
        ] = test_probabilities[
            :,
            index
        ]

    prediction_frame.to_parquet(
        prediction_path,
        index=False,
        compression="zstd"
    )

    model_path = (
        MODEL_DIRECTORY
        / f"{run_name}.pt"
    )

    torch.save(
        {
            "experiment": "ustc_closed_set",
            "seed": arguments.seed,
            "class_names": class_names,
            "normalization": normalization,
            "state_dict": {
                key: value.detach().cpu()
                for (
                    key,
                    value
                ) in model.state_dict().items()
            }
        },
        model_path
    )

    report = {
        "experiment": "ustc_tfc2016_closed_set",
        "model": "cnn_bilstm",
        "seed": arguments.seed,
        "families": class_names,
        "train_rows": len(
            train_frame
        ),
        "validation_rows": len(
            validation_frame
        ),
        "test_rows": len(
            test_frame
        ),
        "training_class_counts": counts.astype(
            int
        ).tolist(),
        "class_weights": weights.astype(
            float
        ).tolist(),
        "normalization": normalization,
        "parameter_count": parameters,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "epochs_completed": len(
            history
        ),
        "training_seconds": training_seconds,
        "validation_batch_inference_seconds": validation_seconds,
        "test_batch_inference_seconds": test_seconds,
        "test_batch_microseconds_per_sample": float(
            test_seconds
            / len(
                test_labels
            )
            * 1000000
        ),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_path": str(
            model_path.relative_to(
                ROOT
            )
        ),
        "model_size_bytes": int(
            model_path.stat().st_size
        ),
        "history_path": str(
            history_path.relative_to(
                ROOT
            )
        ),
        "test_predictions_path": str(
            prediction_path.relative_to(
                ROOT
            )
        )
    }

    report_path = (
        ARTIFACT_DIRECTORY
        / f"{run_name}_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
        + "\n",
        encoding="utf-8"
    )

    print()
    print(
        f"BEST_EPOCH={best_epoch}"
    )

    print(
        f"BEST_VALIDATION_MACRO_F1="
        f"{best_validation_macro_f1:.8f}"
    )

    print(
        f"TEST_ACCURACY="
        f"{test_metrics['accuracy']:.8f}"
    )

    print(
        f"TEST_BALANCED_ACCURACY="
        f"{test_metrics['balanced_accuracy']:.8f}"
    )

    print(
        f"TEST_MACRO_F1="
        f"{test_metrics['macro_f1']:.8f}"
    )

    print(
        f"TEST_WEIGHTED_F1="
        f"{test_metrics['weighted_f1']:.8f}"
    )

    print(
        f"TEST_ROC_AUC_OVR_MACRO="
        f"{test_metrics['roc_auc_ovr_macro']:.8f}"
    )

    print(
        f"TRAINING_SECONDS="
        f"{training_seconds:.3f}"
    )

    print(
        f"TEST_BATCH_US_PER_SAMPLE="
        f"{report['test_batch_microseconds_per_sample']:.6f}"
    )

    print(
        f"PEAK_GPU_MEMORY_MIB="
        f"{peak_gpu_memory_mib:.2f}"
    )

    print(
        f"REPORT={report_path}"
    )

    print(
        "USTC_CLOSED_SET_RUN=PASS"
    )


if __name__ == "__main__":
    main()
