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
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "closed_set"
    / "ustc_closed_model_ready.parquet"
)

PHASE2_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "ustc_protocol.json"
)

SUPPORT_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "ustc_fewshot_support.csv"
)

PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set_protocol.json"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "ustc"
    / "open_set"
)

MODEL_DIRECTORY = (
    ROOT
    / "models"
    / "phase3"
    / "ustc"
    / "open_set"
)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size
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

    def forward(
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

        features = (
            features
            * mask
        )

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

        return torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )


class CEModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes
    ):
        super().__init__()

        self.encoder = Encoder(
            hidden_size
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(
        self,
        inputs,
        lengths
    ):
        embeddings = self.encoder(
            inputs,
            lengths
        )

        return self.classifier(
            self.dropout(
                embeddings
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

        transformed_iats = np.log1p(
            np.asarray(
                iats,
                dtype=np.float64
            )[
                :length
            ]
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
        size_rms <= 0
        or not np.isfinite(
            size_rms
        )
    ):
        raise RuntimeError(
            "Invalid packet-size normalization"
        )

    if (
        iat_std <= 0
        or not np.isfinite(
            iat_std
        )
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


def encode_inputs(
    frame,
    normalization
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
            "Non-finite model inputs"
        )

    return (
        inputs,
        lengths
    )


def labels_for(
    frame,
    mapping
):
    values = (
        frame[
            "class_name"
        ]
        .map(
            mapping
        )
    )

    if values.isna().any():
        raise RuntimeError(
            "Unmapped known-family label"
        )

    return values.to_numpy(
        dtype=np.int64,
        copy=True
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


def make_balanced_loader(
    inputs,
    lengths,
    labels,
    batch_size,
    seed
):
    counts = np.bincount(
        labels
    )

    weights = (
        1.0
        / counts[
            labels
        ]
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            weights,
            dtype=torch.double
        ),
        num_samples=len(
            labels
        ),
        replacement=True,
        generator=generator
    )

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )


def class_weights(
    labels,
    num_classes
):
    counts = np.bincount(
        labels,
        minlength=num_classes
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


def classification_metrics(
    labels,
    predictions,
    class_names
):
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
                len(
                    class_names
                )
            )
        ),
        zero_division=0
    )

    result = {
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
                labels=list(
                    range(
                        len(
                            class_names
                        )
                    )
                ),
                average="macro",
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(
                range(
                    len(
                        class_names
                    )
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

    return result


def embed(
    encoder,
    inputs,
    lengths,
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
        lengths,
        dummy,
        batch_size,
        False,
        0
    )

    batches = []

    encoder.eval()

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

            values = encoder(
                batch_inputs,
                batch_lengths
            )

            values = F.normalize(
                values,
                p=2,
                dim=1
            )

            batches.append(
                values.cpu().numpy()
            )

    return np.concatenate(
        batches,
        axis=0
    )


def build_prototypes(
    embeddings,
    labels,
    num_classes
):
    prototypes = []

    for class_index in range(
        num_classes
    ):
        values = embeddings[
            labels
            == class_index
        ]

        if len(
            values
        ) == 0:
            raise RuntimeError(
                "Prototype class has no embeddings"
            )

        prototype = values.mean(
            axis=0
        )

        norm = np.linalg.norm(
            prototype
        )

        if norm <= 0:
            raise RuntimeError(
                "Zero prototype norm"
            )

        prototypes.append(
            prototype
            / norm
        )

    return np.stack(
        prototypes
    ).astype(
        np.float32
    )


def prototype_predictions(
    embeddings,
    prototypes
):
    similarities = np.clip(
        embeddings
        @ prototypes.T,
        -1.0,
        1.0
    )

    predictions = similarities.argmax(
        axis=1
    )

    maximum_similarity = similarities.max(
        axis=1
    )

    unknown_score = (
        1.0
        - maximum_similarity
    )

    return (
        predictions,
        unknown_score,
        similarities
    )


def predict_ce(
    model,
    inputs,
    lengths,
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
        lengths,
        dummy,
        batch_size,
        False,
        0
    )

    batches = []

    model.eval()

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

            batches.append(
                torch.softmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

    probabilities = np.concatenate(
        batches,
        axis=0
    )

    predictions = probabilities.argmax(
        axis=1
    )

    unknown_score = (
        1.0
        - probabilities.max(
            axis=1
        )
    )

    return (
        predictions,
        unknown_score,
        probabilities
    )


def select_threshold(
    known_scores,
    unknown_scores
):
    labels = np.concatenate(
        (
            np.zeros(
                len(
                    known_scores
                ),
                dtype=np.int64
            ),
            np.ones(
                len(
                    unknown_scores
                ),
                dtype=np.int64
            )
        )
    )

    scores = np.concatenate(
        (
            known_scores,
            unknown_scores
        )
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    finite = np.isfinite(
        thresholds
    )

    fpr = fpr[
        finite
    ]

    tpr = tpr[
        finite
    ]

    thresholds = thresholds[
        finite
    ]

    youden = (
        tpr
        - fpr
    )

    maximum = youden.max()

    candidates = np.flatnonzero(
        np.isclose(
            youden,
            maximum,
            atol=1e-12,
            rtol=0
        )
    )

    best = sorted(
        candidates,
        key=lambda index: (
            -tpr[
                index
            ],
            fpr[
                index
            ],
            thresholds[
                index
            ]
        )
    )[0]

    threshold = float(
        thresholds[
            best
        ]
    )

    return {
        "threshold": threshold,
        "calibration_unknown_recall": float(
            tpr[
                best
            ]
        ),
        "known_false_rejection_rate": float(
            fpr[
                best
            ]
        ),
        "balanced_accuracy": float(
            (
                tpr[
                    best
                ]
                + 1.0
                - fpr[
                    best
                ]
            )
            / 2.0
        )
    }


def zero_shot_metrics(
    known_labels,
    known_predictions,
    known_scores,
    target_predictions,
    target_scores,
    threshold,
    known_names
):
    num_known = len(
        known_names
    )

    binary_labels = np.concatenate(
        (
            np.zeros(
                len(
                    known_scores
                ),
                dtype=np.int64
            ),
            np.ones(
                len(
                    target_scores
                ),
                dtype=np.int64
            )
        )
    )

    binary_scores = np.concatenate(
        (
            known_scores,
            target_scores
        )
    )

    known_rejected = (
        known_scores
        > threshold
    )

    target_rejected = (
        target_scores
        > threshold
    )

    open_true = np.concatenate(
        (
            known_labels,
            np.full(
                len(
                    target_scores
                ),
                num_known,
                dtype=np.int64
            )
        )
    )

    open_predictions = np.concatenate(
        (
            np.where(
                known_rejected,
                num_known,
                known_predictions
            ),
            np.where(
                target_rejected,
                num_known,
                target_predictions
            )
        )
    )

    open_names = (
        list(
            known_names
        )
        + [
            "unknown"
        ]
    )

    return {
        "unknown_detection_auroc": float(
            roc_auc_score(
                binary_labels,
                binary_scores
            )
        ),
        "unknown_detection_average_precision": float(
            average_precision_score(
                binary_labels,
                binary_scores
            )
        ),
        "known_false_rejection_rate": float(
            known_rejected.mean()
        ),
        "known_acceptance_rate": float(
            1.0
            - known_rejected.mean()
        ),
        "unknown_recall": float(
            target_rejected.mean()
        ),
        "unknown_miss_rate": float(
            1.0
            - target_rejected.mean()
        ),
        "known_family_macro_f1_without_rejection": float(
            f1_score(
                known_labels,
                known_predictions,
                average="macro",
                zero_division=0
            )
        ),
        "open_set": classification_metrics(
            open_true,
            open_predictions,
            open_names
        )
    }


def supcon_loss(
    embeddings,
    labels,
    temperature
):
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1
    )

    logits = (
        embeddings
        @ embeddings.T
        / temperature
    )

    batch_size = logits.shape[
        0
    ]

    identity = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=logits.device
    )

    logits = (
        logits
        - logits.max(
            dim=1,
            keepdim=True
        ).values.detach()
    )

    denominator_mask = (
        ~identity
    )

    positive_mask = (
        labels.unsqueeze(
            0
        )
        == labels.unsqueeze(
            1
        )
    ) & denominator_mask

    exponentials = (
        torch.exp(
            logits
        )
        * denominator_mask
    )

    log_probabilities = (
        logits
        - torch.log(
            exponentials.sum(
                dim=1,
                keepdim=True
            )
            + 1e-12
        )
    )

    positive_counts = positive_mask.sum(
        dim=1
    )

    valid = (
        positive_counts
        > 0
    )

    mean_positive_log_probability = (
        (
            positive_mask
            * log_probabilities
        ).sum(
            dim=1
        )[
            valid
        ]
        / positive_counts[
            valid
        ]
    )

    return (
        -mean_positive_log_probability.mean()
    )


def train_ce(
    model,
    train_inputs,
    train_lengths,
    train_labels,
    validation_inputs,
    validation_lengths,
    validation_labels,
    protocol,
    seed,
    device,
    known_names
):
    config = protocol[
        "ce_msp"
    ]

    common = protocol[
        "common_training"
    ]

    (
        counts,
        weights
    ) = class_weights(
        train_labels,
        len(
            known_names
        )
    )

    loader = make_loader(
        train_inputs,
        train_lengths,
        train_labels,
        int(
            config[
                "training_batch_size"
            ]
        ),
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
            common[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            common[
                "weight_decay"
            ]
        )
    )

    best_score = float(
        "-inf"
    )

    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    start = time.perf_counter()

    for epoch in range(
        1,
        int(
            config[
                "maximum_epochs"
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
                    common[
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

        validation_predictions, _, _ = predict_ce(
            model,
            validation_inputs,
            validation_lengths,
            int(
                config[
                    "training_batch_size"
                ]
            ),
            device
        )

        score = float(
            f1_score(
                validation_labels,
                validation_predictions,
                average="macro",
                zero_division=0
            )
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": (
                    total_loss
                    / total_samples
                ),
                "validation_macro_f1": score,
                "best_validation_macro_f1": max(
                    best_score,
                    score
                )
            }
        )

        print(
            f"EPOCH={epoch} "
            f"TRAIN_LOSS="
            f"{total_loss / total_samples:.6f} "
            f"VAL_MACRO_F1={score:.6f} "
            f"BEST={max(best_score, score):.6f}"
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

            stale = 0

        else:
            stale += 1

        if stale >= int(
            config[
                "early_stopping_patience"
            ]
        ):
            print(
                f"EARLY_STOPPING_EPOCH={epoch}"
            )
            break

    training_seconds = (
        time.perf_counter()
        - start
    )

    if best_state is None:
        raise RuntimeError(
            "No CE checkpoint"
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
        training_seconds,
        counts,
        weights
    )


def train_supcon(
    encoder,
    train_inputs,
    train_lengths,
    train_labels,
    validation_inputs,
    validation_lengths,
    validation_labels,
    protocol,
    seed,
    device,
    known_names
):
    config = protocol[
        "supcon_proto"
    ]

    common = protocol[
        "common_training"
    ]

    loader = make_balanced_loader(
        train_inputs,
        train_lengths,
        train_labels,
        int(
            config[
                "training_batch_size"
            ]
        ),
        seed
    )

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(
            common[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            common[
                "weight_decay"
            ]
        )
    )

    best_score = float(
        "-inf"
    )

    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    inference_batch_size = 512

    start = time.perf_counter()

    for epoch in range(
        1,
        int(
            config[
                "maximum_epochs"
            ]
        )
        + 1
    ):
        encoder.train()

        total_loss = 0.0
        total_batches = 0

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

            embeddings = encoder(
                batch_inputs,
                batch_lengths
            )

            loss = supcon_loss(
                embeddings,
                batch_labels,
                float(
                    config[
                        "temperature"
                    ]
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(),
                float(
                    common[
                        "gradient_clip_norm"
                    ]
                )
            )

            optimizer.step()

            total_loss += float(
                loss.detach().item()
            )

            total_batches += 1

        train_embeddings = embed(
            encoder,
            train_inputs,
            train_lengths,
            inference_batch_size,
            device
        )

        prototypes = build_prototypes(
            train_embeddings,
            train_labels,
            len(
                known_names
            )
        )

        validation_embeddings = embed(
            encoder,
            validation_inputs,
            validation_lengths,
            inference_batch_size,
            device
        )

        (
            validation_predictions,
            _,
            _
        ) = prototype_predictions(
            validation_embeddings,
            prototypes
        )

        score = float(
            f1_score(
                validation_labels,
                validation_predictions,
                average="macro",
                zero_division=0
            )
        )

        history.append(
            {
                "epoch": epoch,
                "training_supcon_loss": (
                    total_loss
                    / total_batches
                ),
                "validation_macro_f1": score,
                "best_validation_macro_f1": max(
                    best_score,
                    score
                )
            }
        )

        print(
            f"EPOCH={epoch} "
            f"TRAIN_SUPCON_LOSS="
            f"{total_loss / total_batches:.6f} "
            f"VAL_PROTO_MACRO_F1={score:.6f} "
            f"BEST={max(best_score, score):.6f}"
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
                ) in encoder.state_dict().items()
            }

            stale = 0

        else:
            stale += 1

        if stale >= int(
            config[
                "early_stopping_patience"
            ]
        ):
            print(
                f"EARLY_STOPPING_EPOCH={epoch}"
            )
            break

    training_seconds = (
        time.perf_counter()
        - start
    )

    if best_state is None:
        raise RuntimeError(
            "No SupCon checkpoint"
        )

    encoder.load_state_dict(
        best_state
    )

    encoder.to(
        device
    )

    return (
        encoder,
        history,
        best_epoch,
        best_score,
        training_seconds
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=[
            "ce_msp",
            "supcon_proto"
        ],
        required=True
    )

    parser.add_argument(
        "--run",
        type=int,
        choices=list(
            range(
                10
            )
        ),
        required=True
    )

    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    phase2 = json.loads(
        PHASE2_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    support = pd.read_csv(
        SUPPORT_PATH
    )

    rotation = next(
        item
        for item in phase2[
            "open_set"
        ][
            "runs_detail"
        ]
        if int(
            item[
                "run"
            ]
        )
        == arguments.run
    )

    target_family = rotation[
        "target_unknown_family"
    ]

    calibration_family = rotation[
        "calibration_unknown_family"
    ]

    known_names = list(
        rotation[
            "known_families"
        ]
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
        DATA_PATH
    )

    known_train = frame[
        (
            frame[
                "class_name"
            ].isin(
                known_names
            )
        )
        & (
            frame[
                "closed_split"
            ]
            == "train"
        )
    ].copy()

    known_validation = frame[
        (
            frame[
                "class_name"
            ].isin(
                known_names
            )
        )
        & (
            frame[
                "closed_split"
            ]
            == "validation"
        )
    ].copy()

    known_test = frame[
        (
            frame[
                "class_name"
            ].isin(
                known_names
            )
        )
        & (
            frame[
                "closed_split"
            ]
            == "test"
        )
    ].copy()

    calibration_pool = frame[
        (
            frame[
                "class_name"
            ]
            == calibration_family
        )
        & (
            frame[
                "adaptation_partition"
            ]
            == "calibration_pool"
        )
    ].copy()

    target_support_pool = frame[
        (
            frame[
                "class_name"
            ]
            == target_family
        )
        & (
            frame[
                "adaptation_partition"
            ]
            == "support_pool"
        )
    ].copy()

    target_query = frame[
        (
            frame[
                "class_name"
            ]
            == target_family
        )
        & (
            frame[
                "adaptation_partition"
            ]
            == "query_pool"
        )
    ].copy()

    del frame

    normalization = fit_normalization(
        known_train
    )

    (
        train_inputs,
        train_lengths
    ) = encode_inputs(
        known_train,
        normalization
    )

    (
        validation_inputs,
        validation_lengths
    ) = encode_inputs(
        known_validation,
        normalization
    )

    (
        test_inputs,
        test_lengths
    ) = encode_inputs(
        known_test,
        normalization
    )

    (
        calibration_inputs,
        calibration_lengths
    ) = encode_inputs(
        calibration_pool,
        normalization
    )

    (
        target_query_inputs,
        target_query_lengths
    ) = encode_inputs(
        target_query,
        normalization
    )

    mapping = {
        name: index
        for (
            index,
            name
        ) in enumerate(
            known_names
        )
    }

    train_labels = labels_for(
        known_train,
        mapping
    )

    validation_labels = labels_for(
        known_validation,
        mapping
    )

    test_labels = labels_for(
        known_test,
        mapping
    )

    device = torch.device(
        "cuda:0"
    )

    print(
        f"METHOD={arguments.method}"
    )

    print(
        f"RUN={arguments.run}"
    )

    print(
        f"TARGET_UNKNOWN={target_family}"
    )

    print(
        f"CALIBRATION_UNKNOWN={calibration_family}"
    )

    print(
        f"KNOWN_FAMILIES="
        f"{','.join(known_names)}"
    )

    print(
        f"KNOWN_TRAIN={len(known_train)}"
    )

    print(
        f"KNOWN_VALIDATION={len(known_validation)}"
    )

    print(
        f"KNOWN_TEST={len(known_test)}"
    )

    print(
        f"CALIBRATION_POOL={len(calibration_pool)}"
    )

    print(
        f"TARGET_SUPPORT_POOL={len(target_support_pool)}"
    )

    print(
        f"TARGET_QUERY_POOL={len(target_query)}"
    )

    print(
        f"NORMALIZATION="
        f"{json.dumps(normalization, sort_keys=True)}"
    )

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if arguments.method == "ce_msp":
        model = CEModel(
            hidden_size=int(
                protocol[
                    "encoder"
                ][
                    "bilstm_hidden_size_per_direction"
                ]
            ),
            num_classes=len(
                known_names
            )
        ).to(
            device
        )

        (
            model,
            history,
            best_epoch,
            best_validation_score,
            training_seconds,
            training_counts,
            training_weights
        ) = train_ce(
            model,
            train_inputs,
            train_lengths,
            train_labels,
            validation_inputs,
            validation_lengths,
            validation_labels,
            protocol,
            seed,
            device,
            known_names
        )

        (
            validation_predictions,
            validation_scores,
            _
        ) = predict_ce(
            model,
            validation_inputs,
            validation_lengths,
            512,
            device
        )

        (
            calibration_predictions,
            calibration_scores,
            _
        ) = predict_ce(
            model,
            calibration_inputs,
            calibration_lengths,
            512,
            device
        )

        threshold_report = select_threshold(
            validation_scores,
            calibration_scores
        )

        (
            test_predictions,
            test_scores,
            _
        ) = predict_ce(
            model,
            test_inputs,
            test_lengths,
            512,
            device
        )

        (
            target_predictions,
            target_scores,
            _
        ) = predict_ce(
            model,
            target_query_inputs,
            target_query_lengths,
            512,
            device
        )

        zero_shot = zero_shot_metrics(
            test_labels,
            test_predictions,
            test_scores,
            target_predictions,
            target_scores,
            threshold_report[
                "threshold"
            ],
            known_names
        )

        few_shot = None

        state = {
            "state_dict": {
                key: value.detach().cpu()
                for (
                    key,
                    value
                ) in model.state_dict().items()
            }
        }

    else:
        encoder = Encoder(
            hidden_size=int(
                protocol[
                    "encoder"
                ][
                    "bilstm_hidden_size_per_direction"
                ]
            )
        ).to(
            device
        )

        (
            encoder,
            history,
            best_epoch,
            best_validation_score,
            training_seconds
        ) = train_supcon(
            encoder,
            train_inputs,
            train_lengths,
            train_labels,
            validation_inputs,
            validation_lengths,
            validation_labels,
            protocol,
            seed,
            device,
            known_names
        )

        train_embeddings = embed(
            encoder,
            train_inputs,
            train_lengths,
            512,
            device
        )

        prototypes = build_prototypes(
            train_embeddings,
            train_labels,
            len(
                known_names
            )
        )

        validation_embeddings = embed(
            encoder,
            validation_inputs,
            validation_lengths,
            512,
            device
        )

        (
            validation_predictions,
            validation_scores,
            _
        ) = prototype_predictions(
            validation_embeddings,
            prototypes
        )

        calibration_embeddings = embed(
            encoder,
            calibration_inputs,
            calibration_lengths,
            512,
            device
        )

        (
            calibration_predictions,
            calibration_scores,
            _
        ) = prototype_predictions(
            calibration_embeddings,
            prototypes
        )

        threshold_report = select_threshold(
            validation_scores,
            calibration_scores
        )

        test_embeddings = embed(
            encoder,
            test_inputs,
            test_lengths,
            512,
            device
        )

        (
            test_predictions,
            test_scores,
            _
        ) = prototype_predictions(
            test_embeddings,
            prototypes
        )

        target_query_embeddings = embed(
            encoder,
            target_query_inputs,
            target_query_lengths,
            512,
            device
        )

        (
            target_predictions,
            target_scores,
            _
        ) = prototype_predictions(
            target_query_embeddings,
            prototypes
        )

        zero_shot = zero_shot_metrics(
            test_labels,
            test_predictions,
            test_scores,
            target_predictions,
            target_scores,
            threshold_report[
                "threshold"
            ],
            known_names
        )

        few_shot = {}

        run_support = support[
            support[
                "run"
            ]
            == arguments.run
        ]

        for k in [
            1,
            5,
            10
        ]:
            selected_ids = (
                run_support[
                    run_support[
                        "k"
                    ]
                    == k
                ]
                .sort_values(
                    "support_rank"
                )[
                    "flow_id"
                ]
                .astype(
                    str
                )
                .tolist()
            )

            support_frame = (
                target_support_pool[
                    target_support_pool[
                        "flow_id"
                    ]
                    .astype(
                        str
                    )
                    .isin(
                        selected_ids
                    )
                ]
                .copy()
            )

            if len(
                support_frame
            ) != k:
                raise RuntimeError(
                    f"K={k} support join failure"
                )

            (
                support_inputs,
                support_lengths
            ) = encode_inputs(
                support_frame,
                normalization
            )

            support_embeddings = embed(
                encoder,
                support_inputs,
                support_lengths,
                512,
                device
            )

            target_prototype = support_embeddings.mean(
                axis=0
            )

            target_prototype = (
                target_prototype
                / np.linalg.norm(
                    target_prototype
                )
            )

            adapted_prototypes = np.vstack(
                (
                    prototypes,
                    target_prototype[
                        None,
                        :
                    ]
                )
            ).astype(
                np.float32
            )

            (
                adapted_validation_predictions,
                adapted_validation_scores,
                _
            ) = prototype_predictions(
                validation_embeddings,
                adapted_prototypes
            )

            (
                adapted_calibration_predictions,
                adapted_calibration_scores,
                _
            ) = prototype_predictions(
                calibration_embeddings,
                adapted_prototypes
            )

            adapted_threshold = select_threshold(
                adapted_validation_scores,
                adapted_calibration_scores
            )

            (
                adapted_known_predictions,
                adapted_known_scores,
                _
            ) = prototype_predictions(
                test_embeddings,
                adapted_prototypes
            )

            (
                adapted_target_predictions,
                adapted_target_scores,
                _
            ) = prototype_predictions(
                target_query_embeddings,
                adapted_prototypes
            )

            target_index = len(
                known_names
            )

            combined_true = np.concatenate(
                (
                    test_labels,
                    np.full(
                        len(
                            target_query
                        ),
                        target_index,
                        dtype=np.int64
                    )
                )
            )

            combined_predictions = np.concatenate(
                (
                    adapted_known_predictions,
                    adapted_target_predictions
                )
            )

            adapted_names = (
                known_names
                + [
                    target_family
                ]
            )

            no_rejection = classification_metrics(
                combined_true,
                combined_predictions,
                adapted_names
            )

            unknown_index = (
                target_index
                + 1
            )

            thresholded_predictions = np.concatenate(
                (
                    np.where(
                        adapted_known_scores
                        > adapted_threshold[
                            "threshold"
                        ],
                        unknown_index,
                        adapted_known_predictions
                    ),
                    np.where(
                        adapted_target_scores
                        > adapted_threshold[
                            "threshold"
                        ],
                        unknown_index,
                        adapted_target_predictions
                    )
                )
            )

            thresholded_macro_f1 = float(
                f1_score(
                    combined_true,
                    thresholded_predictions,
                    labels=list(
                        range(
                            len(
                                adapted_names
                            )
                        )
                    ),
                    average="macro",
                    zero_division=0
                )
            )

            few_shot[
                str(
                    k
                )
            ] = {
                "support_flow_ids": selected_ids,
                "threshold": adapted_threshold,
                "nine_family_without_rejection": no_rejection,
                "thresholded_nine_family_macro_f1": thresholded_macro_f1,
                "target_query_acceptance_rate": float(
                    (
                        adapted_target_scores
                        <= adapted_threshold[
                            "threshold"
                        ]
                    ).mean()
                ),
                "target_query_correct_classification_rate": float(
                    (
                        adapted_target_predictions
                        == target_index
                    ).mean()
                ),
                "target_query_correct_and_accepted_rate": float(
                    (
                        (
                            adapted_target_predictions
                            == target_index
                        )
                        & (
                            adapted_target_scores
                            <= adapted_threshold[
                                "threshold"
                            ]
                        )
                    ).mean()
                )
            }

        training_counts = np.bincount(
            train_labels,
            minlength=len(
                known_names
            )
        )

        training_weights = None

        state = {
            "state_dict": {
                key: value.detach().cpu()
                for (
                    key,
                    value
                ) in encoder.state_dict().items()
            },
            "known_prototypes": prototypes
        }

    peak_gpu_memory_mib = float(
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    run_name = (
        f"{arguments.method}_run_{arguments.run}"
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

    model_path = (
        MODEL_DIRECTORY
        / f"{run_name}.pt"
    )

    torch.save(
        {
            "method": arguments.method,
            "run": arguments.run,
            "seed": seed,
            "target_unknown_family": target_family,
            "calibration_unknown_family": calibration_family,
            "known_families": known_names,
            "normalization": normalization,
            **state
        },
        model_path
    )

    report = {
        "experiment": "ustc_tfc2016_open_set",
        "method": arguments.method,
        "run": arguments.run,
        "seed": seed,
        "target_unknown_family": target_family,
        "calibration_unknown_family": calibration_family,
        "known_families": known_names,
        "partition_counts": {
            "known_train": len(
                known_train
            ),
            "known_validation": len(
                known_validation
            ),
            "known_test": len(
                known_test
            ),
            "calibration_pool": len(
                calibration_pool
            ),
            "target_support_pool": len(
                target_support_pool
            ),
            "target_query_pool": len(
                target_query
            )
        },
        "normalization": normalization,
        "training_class_counts": training_counts.astype(
            int
        ).tolist(),
        "training_class_weights": (
            None
            if training_weights is None
            else training_weights.astype(
                float
            ).tolist()
        ),
        "best_epoch": best_epoch,
        "best_known_validation_macro_f1": best_validation_score,
        "epochs_completed": len(
            history
        ),
        "training_seconds": training_seconds,
        "threshold_calibration": threshold_report,
        "zero_shot": zero_shot,
        "few_shot": few_shot,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
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
        f"BEST_KNOWN_VALIDATION_MACRO_F1="
        f"{best_validation_score:.8f}"
    )

    print(
        f"THRESHOLD="
        f"{threshold_report['threshold']:.8f}"
    )

    print(
        f"ZERO_SHOT_AUROC="
        f"{zero_shot['unknown_detection_auroc']:.8f}"
    )

    print(
        f"ZERO_SHOT_AUPR="
        f"{zero_shot['unknown_detection_average_precision']:.8f}"
    )

    print(
        f"ZERO_SHOT_UNKNOWN_RECALL="
        f"{zero_shot['unknown_recall']:.8f}"
    )

    print(
        f"ZERO_SHOT_KNOWN_FRR="
        f"{zero_shot['known_false_rejection_rate']:.8f}"
    )

    print(
        f"ZERO_SHOT_OPEN_MACRO_F1="
        f"{zero_shot['open_set']['macro_f1']:.8f}"
    )

    if few_shot is not None:
        for k in [
            1,
            5,
            10
        ]:
            values = few_shot[
                str(
                    k
                )
            ]

            print(
                f"K{k}_NINE_FAMILY_MACRO_F1="
                f"{values['nine_family_without_rejection']['macro_f1']:.8f}"
            )

            print(
                f"K{k}_TARGET_F1="
                f"{values['nine_family_without_rejection']['per_class'][target_family]['f1']:.8f}"
            )

            print(
                f"K{k}_TARGET_CORRECT_ACCEPTED="
                f"{values['target_query_correct_and_accepted_rate']:.8f}"
            )

    print(
        f"TRAINING_SECONDS="
        f"{training_seconds:.3f}"
    )

    print(
        f"PEAK_GPU_MEMORY_MIB="
        f"{peak_gpu_memory_mib:.2f}"
    )

    print(
        f"REPORT={report_path}"
    )

    print(
        "USTC_OPEN_SET_RUN=PASS"
    )


if __name__ == "__main__":
    main()
