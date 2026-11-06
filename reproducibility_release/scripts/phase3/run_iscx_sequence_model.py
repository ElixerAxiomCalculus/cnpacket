import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

DATASET_PATH = ROOT / "data/processed/iscxvpn2016/iscx_model_ready.parquet"
PHASE2_PROTOCOL_PATH = ROOT / "artifacts/phase2/iscx_model_protocol.json"
PHASE3_PROTOCOL_PATH = ROOT / "artifacts/phase3/iscx/sequence_model_protocol.json"

ARTIFACT_DIRECTORY = ROOT / "artifacts/phase3/iscx/sequence_models"
MODEL_DIRECTORY = ROOT / "models/phase3/iscx/sequence_models"

EXPECTED_RUN_COUNTS = {
    0: {"train_raw": 6242, "train_capped": 5293, "validation": 1830, "test": 2891},
    1: {"train_raw": 7712, "train_capped": 6258, "validation": 1421, "test": 1830},
    2: {"train_raw": 7545, "train_capped": 5882, "validation": 1997, "test": 1421},
    3: {"train_raw": 6142, "train_capped": 5250, "validation": 2824, "test": 1997},
    4: {"train_raw": 5248, "train_capped": 4683, "validation": 2891, "test": 2824}
}


class CNN1D(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs):
        features = self.features(inputs.transpose(1, 2)).squeeze(-1)
        return self.classifier(self.dropout(features))


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size, bidirectional, dropout):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        if self.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]
        return self.classifier(self.dropout(features))


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs):
        features = self.conv(inputs.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(features)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(features))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def load_protocols():
    phase2 = json.loads(PHASE2_PROTOCOL_PATH.read_text(encoding="utf-8"))
    phase3 = json.loads(PHASE3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    return phase2, phase3


def validate_dataset(frame, phase2):
    if len(frame) != 10963:
        raise RuntimeError(f"Expected 10963 rows, found {len(frame)}")

    required = {
        "flow_id",
        "domain",
        "domain_label",
        "activity",
        "activity_label",
        "fold",
        "archive",
        "capture",
        "capture_id",
        "packet_count",
        "sequence_packets",
        "training_key",
        "signed_packet_sizes",
        "interarrival_seconds"
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    activity_mapping = {
        row.activity: int(row.activity_label)
        for row in frame[
            ["activity", "activity_label"]
        ].drop_duplicates().itertuples(index=False)
    }

    domain_mapping = {
        row.domain: int(row.domain_label)
        for row in frame[
            ["domain", "domain_label"]
        ].drop_duplicates().itertuples(index=False)
    }

    if activity_mapping != phase2["activity_labels"]:
        raise RuntimeError("Activity mapping mismatch")

    if domain_mapping != phase2["domain_labels"]:
        raise RuntimeError("Domain mapping mismatch")

    if sorted(frame["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Unexpected fold set")

    if frame["flow_id"].duplicated().any():
        raise RuntimeError("Duplicate flow_id")

    if frame["training_key"].isna().any():
        raise RuntimeError("Missing training_key")


def get_run_config(phase2, run):
    matches = [
        item
        for item in phase2["cross_validation"]["runs"]
        if int(item["run"]) == run
    ]

    if len(matches) != 1:
        raise RuntimeError("Unable to resolve CV run")

    return matches[0]


def split_run(frame, phase2, run):
    config = get_run_config(phase2, run)

    training_folds = [int(value) for value in config["training_folds"]]
    validation_fold = int(config["validation_fold"])
    test_fold = int(config["test_fold"])

    train_raw = frame[frame["fold"].isin(training_folds)].copy()
    validation = frame[frame["fold"] == validation_fold].copy()
    test = frame[frame["fold"] == test_fold].copy()

    train_captures = set(train_raw["capture_id"])
    validation_captures = set(validation["capture_id"])
    test_captures = set(test["capture_id"])

    if train_captures & validation_captures:
        raise RuntimeError("Train/validation capture leakage")

    if train_captures & test_captures:
        raise RuntimeError("Train/test capture leakage")

    if validation_captures & test_captures:
        raise RuntimeError("Validation/test capture leakage")

    cap = int(
        phase2["training_capture_cap"]["maximum_flows_per_capture"]
    )

    train = (
        train_raw
        .sort_values(["capture_id", "training_key"], kind="mergesort")
        .groupby("capture_id", sort=False, group_keys=False)
        .head(cap)
        .copy()
    )

    observed = {
        "train_raw": len(train_raw),
        "train_capped": len(train),
        "validation": len(validation),
        "test": len(test)
    }

    if observed != EXPECTED_RUN_COUNTS[run]:
        raise RuntimeError(
            f"Partition count mismatch: {observed} != {EXPECTED_RUN_COUNTS[run]}"
        )

    if (train.groupby("capture_id").size() > cap).any():
        raise RuntimeError("Training capture cap failure")

    return config, train, validation, test, observed


def extract_sequences(frame, sequence_length):
    sizes = []
    iats = []

    for size_values, iat_values in zip(
        frame["signed_packet_sizes"],
        frame["interarrival_seconds"]
    ):
        size_values = np.asarray(size_values, dtype=np.float32)
        iat_values = np.asarray(iat_values, dtype=np.float32)

        if len(size_values) < sequence_length or len(iat_values) < sequence_length:
            raise RuntimeError("Sequence shorter than frozen N=5 eligibility")

        sizes.append(size_values[:sequence_length])
        iats.append(iat_values[:sequence_length])

    sizes = np.stack(sizes)
    iats = np.stack(iats)

    if not np.isfinite(sizes).all():
        raise RuntimeError("Non-finite packet size")

    if not np.isfinite(iats).all():
        raise RuntimeError("Non-finite interarrival value")

    if (iats < 0).any():
        raise RuntimeError("Negative interarrival value")

    labels = frame["activity_label"].to_numpy(dtype=np.int64, copy=True)

    return sizes, iats, labels


def fit_normalization(sizes, iats):
    signed_size_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    sizes.astype(np.float64)
                )
            )
        )
    )

    transformed_iats = np.log1p(iats.astype(np.float64))

    iat_log1p_mean = float(transformed_iats.mean())
    iat_log1p_std = float(transformed_iats.std(ddof=0))

    if signed_size_rms <= 0 or not np.isfinite(signed_size_rms):
        raise RuntimeError("Invalid packet-size normalization")

    if iat_log1p_std <= 0 or not np.isfinite(iat_log1p_std):
        raise RuntimeError("Invalid IAT normalization")

    return {
        "signed_size_rms": signed_size_rms,
        "iat_log1p_mean": iat_log1p_mean,
        "iat_log1p_std": iat_log1p_std
    }


def transform_sequences(sizes, iats, normalization):
    size_channel = (
        sizes.astype(np.float32)
        / np.float32(normalization["signed_size_rms"])
    )

    iat_channel = (
        (
            np.log1p(iats.astype(np.float64))
            - normalization["iat_log1p_mean"]
        )
        / normalization["iat_log1p_std"]
    ).astype(np.float32)

    inputs = np.stack((size_channel, iat_channel), axis=2)

    if not np.isfinite(inputs).all():
        raise RuntimeError("Non-finite transformed input")

    return inputs


def make_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)

    if (counts == 0).any():
        raise RuntimeError("Training fold missing activity class")

    weights = len(labels) / (num_classes * counts)

    return counts, weights.astype(np.float32)


def make_loader(inputs, labels, batch_size, shuffle, seed, workers):
    dataset = TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(labels)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=worker_seed if workers > 0 else None,
        generator=generator,
        drop_last=False
    )


def build_model(name, protocol, num_classes):
    dropout = float(protocol["training"]["dropout"])

    if name == "cnn":
        return CNN1D(num_classes, dropout)

    if name == "lstm":
        return LSTMClassifier(
            num_classes,
            int(protocol["models"]["lstm"]["hidden_size"]),
            False,
            dropout
        )

    if name == "bilstm":
        return LSTMClassifier(
            num_classes,
            int(protocol["models"]["bilstm"]["hidden_size_per_direction"]),
            True,
            dropout
        )

    if name == "cnn_bilstm":
        return CNNBiLSTM(
            num_classes,
            int(protocol["models"]["cnn_bilstm"]["hidden_size_per_direction"]),
            dropout
        )

    raise RuntimeError(f"Unknown model {name}")


def multiclass_metrics(labels, probabilities, class_names):
    num_classes = len(class_names)
    predictions = probabilities.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(num_classes)),
        zero_division=0
    )

    result = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(num_classes))
        ).astype(int).tolist(),
        "per_class": {}
    }

    for index, name in enumerate(class_names):
        result["per_class"][name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index])
        }

    if len(np.unique(labels)) == num_classes:
        binary = label_binarize(
            labels,
            classes=list(range(num_classes))
        )

        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                labels,
                probabilities,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro"
            )
        )

        result["average_precision_macro"] = float(
            average_precision_score(
                binary,
                probabilities,
                average="macro"
            )
        )
    else:
        result["roc_auc_ovr_macro"] = None
        result["average_precision_macro"] = None

    return result, predictions


def predict_probabilities(
    model,
    inputs,
    batch_size,
    workers,
    device,
    amp
):
    dummy_labels = np.zeros(len(inputs), dtype=np.int64)

    loader = make_loader(
        inputs,
        dummy_labels,
        batch_size,
        False,
        0,
        workers
    )

    batches = []

    model.eval()

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for batch_inputs, _ in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp
            ):
                logits = model(batch_inputs)

            batches.append(
                torch.softmax(logits.float(), dim=1)
                .cpu()
                .numpy()
            )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return np.concatenate(batches, axis=0), elapsed


def train_model(
    model,
    train_inputs,
    train_labels,
    validation_inputs,
    validation_labels,
    class_weights,
    protocol,
    seed,
    device,
    class_names
):
    training = protocol["training"]

    batch_size = int(training["batch_size"])
    workers = int(training["dataloader_workers"])
    amp = bool(training["amp"])

    train_loader = make_loader(
        train_inputs,
        train_labels,
        batch_size,
        True,
        seed,
        workers
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"])
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    torch.cuda.synchronize()
    training_start = time.perf_counter()

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp
            ):
                logits = model(batch_inputs)
                loss = criterion(logits, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            count = len(batch_labels)
            total_loss += float(loss.detach().item()) * count
            total_samples += count

        train_loss = total_loss / total_samples

        validation_probabilities, _ = predict_probabilities(
            model,
            validation_inputs,
            batch_size,
            workers,
            device,
            amp
        )

        validation_metrics, _ = multiclass_metrics(
            validation_labels,
            validation_probabilities,
            class_names
        )

        score = float(validation_metrics["macro_f1"])

        if score > best_score + float(training["early_stopping_min_delta"]):
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
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_weighted_f1": validation_metrics["weighted_f1"],
                "best_macro_f1": best_score
            }
        )

        print(
            f"EPOCH={epoch} "
            f"TRAIN_LOSS={train_loss:.6f} "
            f"VAL_MACRO_F1={score:.6f} "
            f"BEST={best_score:.6f}"
        )

        if stale_epochs >= int(training["early_stopping_patience"]):
            print(f"EARLY_STOPPING_EPOCH={epoch}")
            break

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_start

    if best_state is None:
        raise RuntimeError("No best checkpoint recorded")

    model.load_state_dict(best_state)
    model.to(device)

    return (
        model,
        history,
        best_epoch,
        best_score,
        training_seconds
    )


def domain_metrics(labels, probabilities, domains, class_names):
    results = {}

    for domain_label, domain_name in [(0, "nonvpn"), (1, "vpn")]:
        mask = domains == domain_label

        metrics, _ = multiclass_metrics(
            labels[mask],
            probabilities[mask],
            class_names
        )

        results[domain_name] = metrics

    return results


def parameter_count(model):
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["cnn", "lstm", "bilstm", "cnn_bilstm"],
        required=True
    )

    parser.add_argument(
        "--run",
        type=int,
        choices=[0, 1, 2, 3, 4],
        required=True
    )

    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 3B")

    phase2, phase3 = load_protocols()

    seed = int(phase3["seed"])
    sequence_length = int(phase3["sequence_length"])

    if sequence_length != 5:
        raise RuntimeError("Architecture ladder is frozen to primary N=5")

    set_seed(seed)

    device = torch.device("cuda:0")

    frame = pd.read_parquet(DATASET_PATH)

    validate_dataset(frame, phase2)

    (
        run_config,
        train_frame,
        validation_frame,
        test_frame,
        partition_counts
    ) = split_run(
        frame,
        phase2,
        arguments.run
    )

    del frame

    class_names = [
        name
        for name, label in sorted(
            phase2["activity_labels"].items(),
            key=lambda item: item[1]
        )
    ]

    train_sizes, train_iats, train_labels = extract_sequences(
        train_frame,
        sequence_length
    )

    validation_sizes, validation_iats, validation_labels = extract_sequences(
        validation_frame,
        sequence_length
    )

    test_sizes, test_iats, test_labels = extract_sequences(
        test_frame,
        sequence_length
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

    class_counts, class_weights = make_class_weights(
        train_labels,
        len(class_names)
    )

    model = build_model(
        arguments.model,
        phase3,
        len(class_names)
    ).to(device)

    model_parameters = parameter_count(model)

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(f"MODEL={arguments.model}")
    print(f"RUN={arguments.run}")
    print(f"SEED={seed}")
    print(f"SEQUENCE_LENGTH={sequence_length}")
    print(f"TRAIN_FOLDS={','.join(map(str, run_config['training_folds']))}")
    print(f"VALIDATION_FOLD={run_config['validation_fold']}")
    print(f"TEST_FOLD={run_config['test_fold']}")
    print(f"TRAIN_RAW_ROWS={partition_counts['train_raw']}")
    print(f"TRAIN_CAPPED_ROWS={partition_counts['train_capped']}")
    print(f"VALIDATION_ROWS={partition_counts['validation']}")
    print(f"TEST_ROWS={partition_counts['test']}")
    print(f"PARAMETERS={model_parameters}")
    print(f"TRAIN_CLASS_COUNTS={class_counts.tolist()}")
    print(f"CLASS_WEIGHTS={class_weights.tolist()}")
    print(f"NORMALIZATION={json.dumps(normalization, sort_keys=True)}")

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
        train_labels,
        validation_inputs,
        validation_labels,
        class_weights,
        phase3,
        seed,
        device,
        class_names
    )

    training = phase3["training"]

    validation_probabilities, validation_inference_seconds = (
        predict_probabilities(
            model,
            validation_inputs,
            int(training["batch_size"]),
            int(training["dataloader_workers"]),
            device,
            bool(training["amp"])
        )
    )

    validation_metrics, _ = multiclass_metrics(
        validation_labels,
        validation_probabilities,
        class_names
    )

    test_probabilities, test_inference_seconds = predict_probabilities(
        model,
        test_inputs,
        int(training["batch_size"]),
        int(training["dataloader_workers"]),
        device,
        bool(training["amp"])
    )

    test_metrics, test_predictions = multiclass_metrics(
        test_labels,
        test_probabilities,
        class_names
    )

    test_domains = test_frame["domain_label"].to_numpy(
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

    run_name = f"{arguments.model}_run_{arguments.run}"

    history_path = ARTIFACT_DIRECTORY / f"{run_name}_history.csv"

    pd.DataFrame(history).to_csv(
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
    ].reset_index(drop=True).copy()

    prediction_frame["predicted_label"] = test_predictions

    prediction_frame["predicted_activity"] = [
        class_names[index]
        for index in test_predictions
    ]

    for index, name in enumerate(class_names):
        prediction_frame[f"probability_{name}"] = (
            test_probabilities[:, index]
        )

    prediction_frame.to_parquet(
        prediction_path,
        index=False,
        compression="zstd"
    )

    model_path = MODEL_DIRECTORY / f"{run_name}.pt"

    checkpoint = {
        "model": arguments.model,
        "run": arguments.run,
        "seed": seed,
        "sequence_length": sequence_length,
        "class_names": class_names,
        "normalization": normalization,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
    }

    torch.save(
        checkpoint,
        model_path
    )

    report = {
        "experiment": "iscxvpn2016_sequence_model_ladder",
        "model": arguments.model,
        "run": arguments.run,
        "seed": seed,
        "sequence_length": sequence_length,
        "input_features": [
            "signed_packet_size",
            "interarrival_time"
        ],
        "payload_bytes_used": False,
        "endpoint_addresses_used": False,
        "endpoint_ports_used": False,
        "protocol_identifier_used": False,
        "training_folds": [
            int(value)
            for value in run_config["training_folds"]
        ],
        "validation_fold": int(run_config["validation_fold"]),
        "test_fold": int(run_config["test_fold"]),
        "partition_counts": partition_counts,
        "training_capture_cap": int(
            phase2["training_capture_cap"]["maximum_flows_per_capture"]
        ),
        "normalization": normalization,
        "training_class_counts": class_counts.astype(int).tolist(),
        "class_weights": class_weights.astype(float).tolist(),
        "parameter_count": model_parameters,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "validation_batch_inference_seconds": validation_inference_seconds,
        "validation_batch_microseconds_per_sample": float(
            validation_inference_seconds
            / len(validation_labels)
            * 1000000
        ),
        "test_batch_inference_seconds": test_inference_seconds,
        "test_batch_microseconds_per_sample": float(
            test_inference_seconds
            / len(test_labels)
            * 1000000
        ),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_domain_metrics": test_domain_metrics,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_size_bytes": int(model_path.stat().st_size),
        "history_path": str(history_path.relative_to(ROOT)),
        "test_predictions_path": str(
            prediction_path.relative_to(ROOT)
        )
    }

    report_path = (
        ARTIFACT_DIRECTORY
        / f"{run_name}_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8"
    )

    print()
    print(f"BEST_EPOCH={best_epoch}")
    print(
        f"BEST_VALIDATION_MACRO_F1="
        f"{best_validation_macro_f1:.8f}"
    )
    print(f"TEST_ACCURACY={test_metrics['accuracy']:.8f}")
    print(
        f"TEST_BALANCED_ACCURACY="
        f"{test_metrics['balanced_accuracy']:.8f}"
    )
    print(f"TEST_MACRO_F1={test_metrics['macro_f1']:.8f}")
    print(f"TEST_WEIGHTED_F1={test_metrics['weighted_f1']:.8f}")
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
    print(f"TRAINING_SECONDS={training_seconds:.3f}")
    print(
        f"TEST_BATCH_US_PER_SAMPLE="
        f"{report['test_batch_microseconds_per_sample']:.6f}"
    )
    print(f"PEAK_GPU_MEMORY_MIB={peak_gpu_memory_mib:.2f}")
    print(
        f"MODEL_SIZE_MIB="
        f"{report['model_size_bytes'] / 1024**2:.3f}"
    )
    print(f"REPORT={report_path}")
    print("ISCX_SEQUENCE_MODEL_RUN=PASS")


if __name__ == "__main__":
    main()
