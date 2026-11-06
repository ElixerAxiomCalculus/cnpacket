import argparse
import hashlib
import json
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
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "iscxvpn2016"
    / "iscx_model_ready.parquet"
)

PHASE2_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "iscx_model_protocol.json"
)

PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "iscx"
    / "early_sequence_protocol.json"
)

COHORT_ID_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "iscx"
    / "early_sequence"
    / "ge50_common_cohort_flow_ids.txt"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "iscx"
    / "early_sequence"
)

MODEL_DIRECTORY = (
    ROOT
    / "models"
    / "phase3"
    / "iscx"
    / "early_sequence"
)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size,
        dropout
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                2,
                64,
                3,
                padding=1
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.ReLU()
        )

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

    def forward(
        self,
        inputs
    ):
        features = self.conv(
            inputs.transpose(
                1,
                2
            )
        ).transpose(
            1,
            2
        )

        _, (
            hidden,
            _
        ) = self.lstm(
            features
        )

        features = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        return self.classifier(
            self.dropout(
                features
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


def load_protocols():
    phase2 = json.loads(
        PHASE2_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    return (
        phase2,
        protocol
    )


def cohort_fingerprint(
    flow_ids
):
    payload = (
        "\n".join(
            sorted(
                map(
                    str,
                    flow_ids
                )
            )
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def validate_cohort(
    frame,
    protocol
):
    cohort = frame[
        frame[
            "packet_count"
        ]
        >= 50
    ].copy()

    expected_ids = [
        line.strip()
        for line in COHORT_ID_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    observed_ids = sorted(
        cohort[
            "flow_id"
        ].astype(
            str
        ).tolist()
    )

    if observed_ids != expected_ids:
        raise RuntimeError(
            "GE50 cohort identity mismatch"
        )

    if len(cohort) != int(
        protocol[
            "common_cohort"
        ][
            "flows"
        ]
    ):
        raise RuntimeError(
            "Common cohort row mismatch"
        )

    if cohort[
        "capture_id"
    ].nunique() != int(
        protocol[
            "common_cohort"
        ][
            "captures"
        ]
    ):
        raise RuntimeError(
            "Common cohort capture mismatch"
        )

    expected_folds = {
        int(
            fold
        ): int(
            count
        )
        for fold, count in protocol[
            "common_cohort"
        ][
            "fold_counts"
        ].items()
    }

    observed_folds = {
        int(
            fold
        ): int(
            count
        )
        for fold, count in cohort.groupby(
            "fold"
        ).size().items()
    }

    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Fold count mismatch: "
            f"{observed_folds}"
        )

    for sizes, iats in zip(
        cohort[
            "signed_packet_sizes"
        ],
        cohort[
            "interarrival_seconds"
        ]
    ):
        if len(
            sizes
        ) < 50:
            raise RuntimeError(
                "Stored size sequence shorter than 50"
            )

        if len(
            iats
        ) < 50:
            raise RuntimeError(
                "Stored IAT sequence shorter than 50"
            )

    return cohort


def get_run_config(
    phase2,
    run
):
    matches = [
        item
        for item in phase2[
            "cross_validation"
        ][
            "runs"
        ]
        if int(
            item[
                "run"
            ]
        )
        == run
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Unable to resolve CV run"
        )

    return matches[0]


def split_run(
    cohort,
    phase2,
    protocol,
    run
):
    config = get_run_config(
        phase2,
        run
    )

    train_raw = cohort[
        cohort[
            "fold"
        ].isin(
            config[
                "training_folds"
            ]
        )
    ].copy()

    validation = cohort[
        cohort[
            "fold"
        ]
        == config[
            "validation_fold"
        ]
    ].copy()

    test = cohort[
        cohort[
            "fold"
        ]
        == config[
            "test_fold"
        ]
    ].copy()

    train_captures = set(
        train_raw[
            "capture_id"
        ]
    )

    validation_captures = set(
        validation[
            "capture_id"
        ]
    )

    test_captures = set(
        test[
            "capture_id"
        ]
    )

    if (
        train_captures
        & validation_captures
    ):
        raise RuntimeError(
            "Train/validation capture leakage"
        )

    if (
        train_captures
        & test_captures
    ):
        raise RuntimeError(
            "Train/test capture leakage"
        )

    if (
        validation_captures
        & test_captures
    ):
        raise RuntimeError(
            "Validation/test capture leakage"
        )

    cap = int(
        phase2[
            "training_capture_cap"
        ][
            "maximum_flows_per_capture"
        ]
    )

    train = (
        train_raw
        .sort_values(
            [
                "capture_id",
                "training_key"
            ],
            kind="mergesort"
        )
        .groupby(
            "capture_id",
            sort=False,
            group_keys=False
        )
        .head(
            cap
        )
        .copy()
    )

    observed = {
        "train_raw": len(
            train_raw
        ),
        "train_capped": len(
            train
        ),
        "validation": len(
            validation
        ),
        "test": len(
            test
        )
    }

    expected = {
        key: int(
            value
        )
        for key, value in protocol[
            "run_counts"
        ][
            str(
                run
            )
        ].items()
    }

    if observed != expected:
        raise RuntimeError(
            f"Partition mismatch: "
            f"{observed} != {expected}"
        )

    if len(
        train
    ) != len(
        train_raw
    ):
        raise RuntimeError(
            "Training cap unexpectedly active "
            "on GE50 cohort"
        )

    return (
        config,
        train,
        validation,
        test,
        observed
    )


def extract_sequences(
    frame,
    sequence_length
):
    size_rows = []
    iat_rows = []

    for (
        sizes,
        iats
    ) in zip(
        frame[
            "signed_packet_sizes"
        ],
        frame[
            "interarrival_seconds"
        ]
    ):
        sizes = np.asarray(
            sizes,
            dtype=np.float32
        )

        iats = np.asarray(
            iats,
            dtype=np.float32
        )

        if len(
            sizes
        ) < sequence_length:
            raise RuntimeError(
                "Packet sequence too short"
            )

        if len(
            iats
        ) < sequence_length:
            raise RuntimeError(
                "IAT sequence too short"
            )

        size_rows.append(
            sizes[
                :sequence_length
            ]
        )

        iat_rows.append(
            iats[
                :sequence_length
            ]
        )

    sizes = np.stack(
        size_rows
    )

    iats = np.stack(
        iat_rows
    )

    if not np.isfinite(
        sizes
    ).all():
        raise RuntimeError(
            "Non-finite packet sizes"
        )

    if not np.isfinite(
        iats
    ).all():
        raise RuntimeError(
            "Non-finite IATs"
        )

    if (
        iats
        < 0
    ).any():
        raise RuntimeError(
            "Negative IAT"
        )

    labels = frame[
        "activity_label"
    ].to_numpy(
        dtype=np.int64,
        copy=True
    )

    return (
        sizes,
        iats,
        labels
    )


def fit_normalization(
    sizes,
    iats
):
    signed_size_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    sizes.astype(
                        np.float64
                    )
                )
            )
        )
    )

    transformed_iats = np.log1p(
        iats.astype(
            np.float64
        )
    )

    iat_mean = float(
        transformed_iats.mean()
    )

    iat_std = float(
        transformed_iats.std(
            ddof=0
        )
    )

    if (
        signed_size_rms
        <= 0
        or not np.isfinite(
            signed_size_rms
        )
    ):
        raise RuntimeError(
            "Invalid packet-size normalization"
        )

    if (
        iat_std
        <= 0
        or not np.isfinite(
            iat_std
        )
    ):
        raise RuntimeError(
            "Invalid IAT normalization"
        )

    return {
        "signed_size_rms": signed_size_rms,
        "iat_log1p_mean": iat_mean,
        "iat_log1p_std": iat_std
    }


def transform_sequences(
    sizes,
    iats,
    normalization
):
    size_channel = (
        sizes.astype(
            np.float32
        )
        / np.float32(
            normalization[
                "signed_size_rms"
            ]
        )
    )

    iat_channel = (
        (
            np.log1p(
                iats.astype(
                    np.float64
                )
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

    inputs = np.stack(
        (
            size_channel,
            iat_channel
        ),
        axis=2
    )

    if not np.isfinite(
        inputs
    ).all():
        raise RuntimeError(
            "Non-finite transformed input"
        )

    return inputs


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
            "Training partition missing a class"
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


def make_loader(
    inputs,
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
        generator=generator,
        drop_last=False
    )


def metrics(
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

    for index, name in enumerate(
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

    if len(
        np.unique(
            labels
        )
    ) == num_classes:
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

    else:
        result[
            "roc_auc_ovr_macro"
        ] = None

        result[
            "average_precision_macro"
        ] = None

    return (
        result,
        predictions
    )


def predict(
    model,
    inputs,
    batch_size,
    device
):
    dummy = np.zeros(
        len(
            inputs
        ),
        dtype=np.int64
    )

    loader = make_loader(
        inputs,
        dummy,
        batch_size,
        False,
        0
    )

    batches = []

    model.eval()

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        for (
            batch_inputs,
            _
        ) in loader:
            batch_inputs = batch_inputs.to(
                device,
                non_blocking=True
            )

            logits = model(
                batch_inputs
            )

            batches.append(
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
            batches,
            axis=0
        ),
        elapsed
    )


def train(
    model,
    train_inputs,
    train_labels,
    validation_inputs,
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

    start = time.perf_counter()

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
                batch_inputs
            )

            loss = criterion(
                logits,
                batch_labels
            )

            loss.backward()

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
            batch_size,
            device
        )

        validation_metrics, _ = metrics(
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
                for key, value in model.state_dict().items()
            }

            stale_epochs = 0

        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_macro_f1": score,
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
                f"EARLY_STOPPING_EPOCH="
                f"{epoch}"
            )

            break

    torch.cuda.synchronize()

    training_seconds = (
        time.perf_counter()
        - start
    )

    if best_state is None:
        raise RuntimeError(
            "No best model checkpoint"
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


def domain_metrics(
    labels,
    probabilities,
    domains,
    class_names
):
    result = {}

    for (
        domain_label,
        domain_name
    ) in [
        (
            0,
            "nonvpn"
        ),
        (
            1,
            "vpn"
        )
    ]:
        mask = (
            domains
            == domain_label
        )

        values, _ = metrics(
            labels[
                mask
            ],
            probabilities[
                mask
            ],
            class_names
        )

        result[
            domain_name
        ] = values

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=[
            5,
            10,
            20,
            50
        ],
        required=True
    )

    parser.add_argument(
        "--run",
        type=int,
        choices=[
            0,
            1,
            2,
            3,
            4
        ],
        required=True
    )

    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    (
        phase2,
        protocol
    ) = load_protocols()

    if arguments.sequence_length not in protocol[
        "sequence_lengths"
    ]:
        raise RuntimeError(
            "Sequence length outside frozen protocol"
        )

    seed = int(
        protocol[
            "seed"
        ]
    )

    set_seed(
        seed
    )

    frame = pd.read_parquet(
        DATASET_PATH
    )

    cohort = validate_cohort(
        frame,
        protocol
    )

    del frame

    (
        run_config,
        train_frame,
        validation_frame,
        test_frame,
        partition_counts
    ) = split_run(
        cohort,
        phase2,
        protocol,
        arguments.run
    )

    cohort_hash = cohort_fingerprint(
        cohort[
            "flow_id"
        ]
    )

    class_names = [
        name
        for (
            name,
            label
        ) in sorted(
            phase2[
                "activity_labels"
            ].items(),
            key=lambda item: item[
                1
            ]
        )
    ]

    (
        train_sizes,
        train_iats,
        train_labels
    ) = extract_sequences(
        train_frame,
        arguments.sequence_length
    )

    (
        validation_sizes,
        validation_iats,
        validation_labels
    ) = extract_sequences(
        validation_frame,
        arguments.sequence_length
    )

    (
        test_sizes,
        test_iats,
        test_labels
    ) = extract_sequences(
        test_frame,
        arguments.sequence_length
    )

    normalization = fit_normalization(
        train_sizes,
        train_iats
    )

    train_inputs = transform_sequences(
        train_sizes,
        train_iats,
        normalization
    )

    validation_inputs = transform_sequences(
        validation_sizes,
        validation_iats,
        normalization
    )

    test_inputs = transform_sequences(
        test_sizes,
        test_iats,
        normalization
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
    )

    device = torch.device(
        "cuda:0"
    )

    model = model.to(
        device
    )

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "MODEL=cnn_bilstm"
    )

    print(
        f"SEQUENCE_LENGTH="
        f"{arguments.sequence_length}"
    )

    print(
        f"RUN={arguments.run}"
    )

    print(
        f"SEED={seed}"
    )

    print(
        f"COHORT_ROWS="
        f"{len(cohort)}"
    )

    print(
        f"COHORT_SHA256="
        f"{cohort_hash}"
    )

    print(
        f"TRAIN_FOLDS="
        f"{','.join(map(str, run_config['training_folds']))}"
    )

    print(
        f"VALIDATION_FOLD="
        f"{run_config['validation_fold']}"
    )

    print(
        f"TEST_FOLD="
        f"{run_config['test_fold']}"
    )

    print(
        f"TRAIN_RAW_ROWS="
        f"{partition_counts['train_raw']}"
    )

    print(
        f"TRAIN_CAPPED_ROWS="
        f"{partition_counts['train_capped']}"
    )

    print(
        f"VALIDATION_ROWS="
        f"{partition_counts['validation']}"
    )

    print(
        f"TEST_ROWS="
        f"{partition_counts['test']}"
    )

    print(
        f"TRAIN_CLASS_COUNTS="
        f"{counts.tolist()}"
    )

    print(
        f"PARAMETERS="
        f"{parameter_count}"
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
    ) = train(
        model,
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        weights,
        protocol,
        seed,
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
        batch_size,
        device
    )

    (
        validation_metrics,
        _
    ) = metrics(
        validation_labels,
        validation_probabilities,
        class_names
    )

    test_probabilities, test_seconds = predict(
        model,
        test_inputs,
        batch_size,
        device
    )

    (
        test_metrics,
        test_predictions
    ) = metrics(
        test_labels,
        test_probabilities,
        class_names
    )

    test_domains = test_frame[
        "domain_label"
    ].to_numpy(
        dtype=np.int64,
        copy=True
    )

    test_domain_metrics = domain_metrics(
        test_labels,
        test_probabilities,
        test_domains,
        class_names
    )

    peak_gpu_memory_mib = float(
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    run_name = (
        f"cnn_bilstm_"
        f"n{arguments.sequence_length}_"
        f"run_{arguments.run}"
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
            "capture_id",
            "domain",
            "domain_label",
            "activity",
            "activity_label",
            "fold"
        ]
    ].reset_index(
        drop=True
    ).copy()

    prediction_frame[
        "predicted_label"
    ] = test_predictions

    prediction_frame[
        "predicted_activity"
    ] = [
        class_names[
            index
        ]
        for index in test_predictions
    ]

    for (
        index,
        name
    ) in enumerate(
        class_names
    ):
        prediction_frame[
            f"probability_{name}"
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
            "model": "cnn_bilstm",
            "sequence_length": arguments.sequence_length,
            "run": arguments.run,
            "seed": seed,
            "class_names": class_names,
            "normalization": normalization,
            "cohort_sha256": cohort_hash,
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
        "experiment": "iscxvpn2016_early_classification",
        "model": "cnn_bilstm",
        "sequence_length": arguments.sequence_length,
        "run": arguments.run,
        "seed": seed,
        "cohort_requirement": "packet_count >= 50",
        "cohort_flows": len(
            cohort
        ),
        "cohort_captures": int(
            cohort[
                "capture_id"
            ].nunique()
        ),
        "cohort_sha256": cohort_hash,
        "training_folds": [
            int(
                value
            )
            for value in run_config[
                "training_folds"
            ]
        ],
        "validation_fold": int(
            run_config[
                "validation_fold"
            ]
        ),
        "test_fold": int(
            run_config[
                "test_fold"
            ]
        ),
        "partition_counts": partition_counts,
        "training_cap_active": False,
        "normalization": normalization,
        "training_class_counts": counts.astype(
            int
        ).tolist(),
        "class_weights": weights.astype(
            float
        ).tolist(),
        "parameter_count": parameter_count,
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
        "test_domain_metrics": test_domain_metrics,
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
        f"BEST_EPOCH="
        f"{best_epoch}"
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
        f"TEST_ROC_AUC_OVR_MACRO="
        f"{test_metrics['roc_auc_ovr_macro']:.8f}"
    )

    print(
        f"NONVPN_MACRO_F1="
        f"{test_domain_metrics['nonvpn']['macro_f1']:.8f}"
    )

    print(
        f"VPN_MACRO_F1="
        f"{test_domain_metrics['vpn']['macro_f1']:.8f}"
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
        f"REPORT="
        f"{report_path}"
    )

    print(
        "ISCX_EARLY_SEQUENCE_RUN=PASS"
    )


if __name__ == "__main__":
    main()
