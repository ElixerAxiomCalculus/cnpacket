import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "cicids2017"
    / "cicids2017_clean.parquet"
)

COLUMN_MAPPING_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "cicids_column_mapping.json"
)

PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase3"
    / "cicids"
    / "tree_final_protocol.json"
)

ARTIFACT_DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase3"
    / "cicids"
    / "final"
)

MODEL_DIRECTORY = (
    ROOT
    / "models"
    / "phase3"
    / "cicids"
    / "final"
)

EXPECTED = {
    "train": {
        "rows": 1763582,
        "benign": 1466052,
        "attack": 297530
    },
    "validation": {
        "rows": 377911,
        "benign": 314154,
        "attack": 63757
    },
    "test": {
        "rows": 377911,
        "benign": 314154,
        "attack": 63757
    }
}


def release_memory():
    gc.collect()

    try:
        ctypes.CDLL(
            "libc.so.6"
        ).malloc_trim(
            0
        )
    except Exception:
        pass


def memory_values():
    values = {}

    for line in Path(
        "/proc/self/status"
    ).read_text().splitlines():
        if line.startswith(
            "VmRSS:"
        ):
            values[
                "rss_mib"
            ] = int(
                line.split()[1]
            ) / 1024

        elif line.startswith(
            "VmHWM:"
        ):
            values[
                "peak_mib"
            ] = int(
                line.split()[1]
            ) / 1024

    return values


def print_memory(
    label
):
    values = memory_values()

    print(
        f"MEMORY_{label}_RSS_MIB="
        f"{values.get('rss_mib', 0):.1f}"
    )

    print(
        f"MEMORY_{label}_PEAK_MIB="
        f"{values.get('peak_mib', 0):.1f}"
    )


def feature_names():
    mapping = json.loads(
        COLUMN_MAPPING_PATH.read_text(
            encoding="utf-8"
        )
    )

    features = [
        row["clean"]
        for row in mapping
        if row["clean"] != "label"
    ]

    if len(features) != 78:
        raise RuntimeError(
            f"Expected 78 features, "
            f"found {len(features)}"
        )

    if len(
        set(
            features
        )
    ) != 78:
        raise RuntimeError(
            "Feature names are not unique"
        )

    return features


def validate_finite(
    matrix,
    split
):
    size = 100000

    for start in range(
        0,
        len(matrix),
        size
    ):
        stop = min(
            start + size,
            len(matrix)
        )

        if not np.isfinite(
            matrix[
                start:stop
            ]
        ).all():
            raise RuntimeError(
                f"Non-finite values in {split}"
            )


def load_partition(
    dataset,
    features,
    split,
    row_ids_required
):
    columns = (
        features
        + [
            "binary_label"
        ]
    )

    if row_ids_required:
        columns.append(
            "row_id"
        )

    print(
        f"LOADING_{split.upper()}"
    )

    table = dataset.to_table(
        columns=columns,
        filter=(
            ds.field(
                "benchmark_split"
            )
            == split
        ),
        use_threads=False
    )

    frame = table.to_pandas()

    del table
    release_memory()

    labels = frame.pop(
        "binary_label"
    ).to_numpy(
        dtype=np.int8,
        copy=True
    )

    if row_ids_required:
        row_ids = frame.pop(
            "row_id"
        ).to_numpy(
            copy=True
        )
    else:
        row_ids = None

    matrix = frame.to_numpy(
        dtype=np.float32,
        copy=True
    )

    del frame
    release_memory()

    expected = EXPECTED[
        split
    ]

    benign = int(
        (
            labels
            == 0
        ).sum()
    )

    attack = int(
        (
            labels
            == 1
        ).sum()
    )

    if len(labels) != expected[
        "rows"
    ]:
        raise RuntimeError(
            f"{split} row mismatch"
        )

    if benign != expected[
        "benign"
    ]:
        raise RuntimeError(
            f"{split} benign mismatch"
        )

    if attack != expected[
        "attack"
    ]:
        raise RuntimeError(
            f"{split} attack mismatch"
        )

    validate_finite(
        matrix,
        split
    )

    print(
        f"{split.upper()}_ROWS="
        f"{len(labels)}"
    )

    print(
        f"{split.upper()}_BENIGN="
        f"{benign}"
    )

    print(
        f"{split.upper()}_ATTACK="
        f"{attack}"
    )

    print_memory(
        f"AFTER_{split.upper()}_LOAD"
    )

    return (
        matrix,
        labels,
        row_ids
    )


def metric_report(
    labels,
    probabilities,
    threshold
):
    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int8
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1
        ]
    )

    tn = int(
        matrix[0, 0]
    )

    fp = int(
        matrix[0, 1]
    )

    fn = int(
        matrix[1, 0]
    )

    tp = int(
        matrix[1, 1]
    )

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[
            0,
            1
        ],
        zero_division=0
    )

    return {
        "samples": int(
            len(labels)
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
        "precision_benign": float(
            precision[0]
        ),
        "recall_benign": float(
            recall[0]
        ),
        "f1_benign": float(
            f1[0]
        ),
        "precision_attack": float(
            precision[1]
        ),
        "recall_attack": float(
            recall[1]
        ),
        "f1_attack": float(
            f1[1]
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
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities
            )
        ),
        "false_positive_rate": float(
            fp
            / (
                fp + tn
            )
        ),
        "false_negative_rate": float(
            fn
            / (
                fn + tp
            )
        ),
        "support_benign": int(
            support[0]
        ),
        "support_attack": int(
            support[1]
        ),
        "confusion_matrix": [
            [
                tn,
                fp
            ],
            [
                fn,
                tp
            ]
        ]
    }


def save_predictions(
    model_name,
    seed,
    split,
    row_ids,
    labels,
    probabilities,
    threshold
):
    frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "binary_label": labels,
            "probability_attack": probabilities,
            "prediction": (
                probabilities
                >= threshold
            ).astype(
                np.int8
            )
        }
    )

    path = (
        ARTIFACT_DIRECTORY
        / (
            f"{model_name}_seed_{seed}_"
            f"{split}_predictions.parquet"
        )
    )

    frame.to_parquet(
        path,
        index=False,
        compression="zstd"
    )

    del frame
    release_memory()

    return path


def evaluate_random_forest(
    model,
    model_name,
    seed,
    split,
    dataset,
    features,
    threshold
):
    (
        matrix,
        labels,
        row_ids
    ) = load_partition(
        dataset,
        features,
        split,
        True
    )

    start = time.perf_counter()

    probabilities = (
        model.predict_proba(
            matrix
        )[:, 1]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    metrics = metric_report(
        labels,
        probabilities,
        threshold
    )

    path = save_predictions(
        model_name,
        seed,
        split,
        row_ids,
        labels,
        probabilities,
        threshold
    )

    count = len(
        labels
    )

    del matrix
    del labels
    del row_ids
    del probabilities

    release_memory()

    return {
        "metrics": metrics,
        "prediction_seconds": float(
            elapsed
        ),
        "samples": int(
            count
        ),
        "predictions": str(
            path.relative_to(
                ROOT
            )
        )
    }


def run_random_forest(
    protocol,
    dataset,
    features,
    seed,
    threads,
    threshold
):
    configuration = protocol[
        "random_forest"
    ]

    (
        train_matrix,
        train_labels,
        _
    ) = load_partition(
        dataset,
        features,
        "train",
        False
    )

    model = RandomForestClassifier(
        n_estimators=configuration[
            "n_estimators"
        ],
        criterion=configuration[
            "criterion"
        ],
        max_depth=configuration[
            "max_depth"
        ],
        min_samples_leaf=configuration[
            "min_samples_leaf"
        ],
        max_features=configuration[
            "max_features"
        ],
        max_samples=configuration[
            "max_samples"
        ],
        bootstrap=configuration[
            "bootstrap"
        ],
        class_weight=configuration[
            "class_weight"
        ],
        n_jobs=threads,
        random_state=seed,
        verbose=0
    )

    print(
        "TRAINING_BEGIN"
    )

    start = time.perf_counter()

    model.fit(
        train_matrix,
        train_labels
    )

    training_seconds = (
        time.perf_counter()
        - start
    )

    print(
        "TRAINING_COMPLETE"
    )

    print_memory(
        "AFTER_TRAINING"
    )

    del train_matrix
    del train_labels
    release_memory()

    model_path = (
        MODEL_DIRECTORY
        / f"random_forest_seed_{seed}.joblib"
    )

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    validation = evaluate_random_forest(
        model,
        "random_forest",
        seed,
        "validation",
        dataset,
        features,
        threshold
    )

    test = evaluate_random_forest(
        model,
        "random_forest",
        seed,
        "test",
        dataset,
        features,
        threshold
    )

    return {
        "model": "random_forest",
        "seed": seed,
        "best_iteration": None,
        "training_seconds": float(
            training_seconds
        ),
        "model_path": str(
            model_path.relative_to(
                ROOT
            )
        ),
        "model_size_bytes": int(
            model_path.stat().st_size
        ),
        "validation": validation,
        "test": test
    }


def run_xgboost(
    protocol,
    dataset,
    features,
    seed,
    threads,
    threshold
):
    configuration = protocol[
        "xgboost"
    ]

    (
        train_matrix,
        train_labels,
        _
    ) = load_partition(
        dataset,
        features,
        "train",
        False
    )

    negatives = int(
        (
            train_labels
            == 0
        ).sum()
    )

    positives = int(
        (
            train_labels
            == 1
        ).sum()
    )

    scale_pos_weight = (
        negatives
        / positives
    )

    train_dmatrix = xgb.QuantileDMatrix(
        train_matrix,
        label=train_labels,
        feature_names=features,
        max_bin=configuration[
            "max_bin"
        ]
    )

    del train_matrix
    del train_labels
    release_memory()

    (
        validation_matrix,
        validation_labels,
        validation_row_ids
    ) = load_partition(
        dataset,
        features,
        "validation",
        True
    )

    validation_dmatrix = (
        xgb.QuantileDMatrix(
            validation_matrix,
            label=validation_labels,
            feature_names=features,
            max_bin=configuration[
                "max_bin"
            ],
            ref=train_dmatrix
        )
    )

    del validation_matrix
    release_memory()

    parameters = {
        "objective": configuration[
            "objective"
        ],
        "eval_metric": configuration[
            "eval_metric"
        ],
        "tree_method": configuration[
            "tree_method"
        ],
        "max_bin": configuration[
            "max_bin"
        ],
        "max_depth": configuration[
            "max_depth"
        ],
        "eta": configuration[
            "eta"
        ],
        "subsample": configuration[
            "subsample"
        ],
        "colsample_bytree": configuration[
            "colsample_bytree"
        ],
        "min_child_weight": configuration[
            "min_child_weight"
        ],
        "lambda": configuration[
            "lambda"
        ],
        "alpha": configuration[
            "alpha"
        ],
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "seed": seed,
        "nthread": threads
    }

    print(
        f"SCALE_POS_WEIGHT="
        f"{scale_pos_weight:.8f}"
    )

    print_memory(
        "BEFORE_TRAINING"
    )

    print(
        "TRAINING_BEGIN"
    )

    start = time.perf_counter()

    model = xgb.train(
        parameters,
        train_dmatrix,
        num_boost_round=configuration[
            "num_boost_round"
        ],
        evals=[
            (
                train_dmatrix,
                "train"
            ),
            (
                validation_dmatrix,
                "validation"
            )
        ],
        early_stopping_rounds=configuration[
            "early_stopping_rounds"
        ],
        verbose_eval=50
    )

    training_seconds = (
        time.perf_counter()
        - start
    )

    print(
        "TRAINING_COMPLETE"
    )

    best_iteration = int(
        model.best_iteration
    )

    validation_start = (
        time.perf_counter()
    )

    validation_probabilities = (
        model.predict(
            validation_dmatrix,
            iteration_range=(
                0,
                best_iteration + 1
            )
        )
    )

    validation_seconds = (
        time.perf_counter()
        - validation_start
    )

    validation_metrics = metric_report(
        validation_labels,
        validation_probabilities,
        threshold
    )

    validation_path = save_predictions(
        "xgboost",
        seed,
        "validation",
        validation_row_ids,
        validation_labels,
        validation_probabilities,
        threshold
    )

    del validation_probabilities
    del validation_labels
    del validation_row_ids
    del validation_dmatrix
    del train_dmatrix

    release_memory()

    (
        test_matrix,
        test_labels,
        test_row_ids
    ) = load_partition(
        dataset,
        features,
        "test",
        True
    )

    test_dmatrix = xgb.DMatrix(
        test_matrix,
        label=test_labels,
        feature_names=features
    )

    del test_matrix
    release_memory()

    test_start = (
        time.perf_counter()
    )

    test_probabilities = (
        model.predict(
            test_dmatrix,
            iteration_range=(
                0,
                best_iteration + 1
            )
        )
    )

    test_seconds = (
        time.perf_counter()
        - test_start
    )

    test_metrics = metric_report(
        test_labels,
        test_probabilities,
        threshold
    )

    test_path = save_predictions(
        "xgboost",
        seed,
        "test",
        test_row_ids,
        test_labels,
        test_probabilities,
        threshold
    )

    del test_dmatrix
    del test_probabilities
    del test_labels
    del test_row_ids

    release_memory()

    model_path = (
        MODEL_DIRECTORY
        / f"xgboost_seed_{seed}.ubj"
    )

    model.save_model(
        model_path
    )

    return {
        "model": "xgboost",
        "seed": seed,
        "best_iteration": (
            best_iteration
        ),
        "training_seconds": float(
            training_seconds
        ),
        "model_path": str(
            model_path.relative_to(
                ROOT
            )
        ),
        "model_size_bytes": int(
            model_path.stat().st_size
        ),
        "validation": {
            "metrics": (
                validation_metrics
            ),
            "prediction_seconds": float(
                validation_seconds
            ),
            "samples": EXPECTED[
                "validation"
            ][
                "rows"
            ],
            "predictions": str(
                validation_path.relative_to(
                    ROOT
                )
            )
        },
        "test": {
            "metrics": (
                test_metrics
            ),
            "prediction_seconds": float(
                test_seconds
            ),
            "samples": EXPECTED[
                "test"
            ][
                "rows"
            ],
            "predictions": str(
                test_path.relative_to(
                    ROOT
                )
            )
        }
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "rf",
            "xgb"
        ],
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2
    )

    arguments = parser.parse_args()

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if arguments.seed not in protocol[
        "seeds"
    ]:
        raise RuntimeError(
            f"Seed {arguments.seed} "
            f"is not frozen in protocol"
        )

    if arguments.threads != protocol[
        "threads"
    ]:
        raise RuntimeError(
            f"Expected "
            f"{protocol['threads']} threads"
        )

    threshold = float(
        protocol[
            "threshold"
        ]
    )

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    features = feature_names()

    dataset = ds.dataset(
        str(
            DATASET_PATH
        ),
        format="parquet"
    )

    print(
        f"MODEL={arguments.model}"
    )

    print(
        f"SEED={arguments.seed}"
    )

    print(
        f"THREADS={arguments.threads}"
    )

    print(
        f"THRESHOLD={threshold}"
    )

    print(
        f"FEATURES={len(features)}"
    )

    print_memory(
        "START"
    )

    if arguments.model == "rf":
        result = run_random_forest(
            protocol,
            dataset,
            features,
            arguments.seed,
            arguments.threads,
            threshold
        )

    else:
        result = run_xgboost(
            protocol,
            dataset,
            features,
            arguments.seed,
            arguments.threads,
            threshold
        )

    test_seconds = result[
        "test"
    ][
        "prediction_seconds"
    ]

    test_samples = result[
        "test"
    ][
        "samples"
    ]

    result[
        "test_microseconds_per_sample"
    ] = float(
        test_seconds
        / test_samples
        * 1000000
    )

    result[
        "test_samples_per_second"
    ] = float(
        test_samples
        / test_seconds
    )

    result[
        "protocol"
    ] = str(
        PROTOCOL_PATH.relative_to(
            ROOT
        )
    )

    result[
        "threshold"
    ] = threshold

    report_path = (
        ARTIFACT_DIRECTORY
        / (
            f"{result['model']}_"
            f"seed_{arguments.seed}_report.json"
        )
    )

    report_path.write_text(
        json.dumps(
            result,
            indent=2
        )
        + "\n",
        encoding="utf-8"
    )

    test_metrics = result[
        "test"
    ][
        "metrics"
    ]

    validation_metrics = result[
        "validation"
    ][
        "metrics"
    ]

    print()
    print(
        f"MODEL_NAME={result['model']}"
    )

    print(
        f"SEED={arguments.seed}"
    )

    print(
        f"TRAINING_SECONDS="
        f"{result['training_seconds']:.3f}"
    )

    if result[
        "best_iteration"
    ] is not None:
        print(
            f"BEST_ITERATION="
            f"{result['best_iteration']}"
        )

    print(
        f"VALIDATION_MACRO_F1="
        f"{validation_metrics['macro_f1']:.8f}"
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
        f"TEST_ATTACK_PRECISION="
        f"{test_metrics['precision_attack']:.8f}"
    )

    print(
        f"TEST_ATTACK_RECALL="
        f"{test_metrics['recall_attack']:.8f}"
    )

    print(
        f"TEST_ATTACK_F1="
        f"{test_metrics['f1_attack']:.8f}"
    )

    print(
        f"TEST_ROC_AUC="
        f"{test_metrics['roc_auc']:.8f}"
    )

    print(
        f"TEST_AVERAGE_PRECISION="
        f"{test_metrics['average_precision']:.8f}"
    )

    print(
        f"TEST_FPR="
        f"{test_metrics['false_positive_rate']:.8f}"
    )

    print(
        f"TEST_FNR="
        f"{test_metrics['false_negative_rate']:.8f}"
    )

    print(
        f"TEST_CONFUSION_MATRIX="
        f"{test_metrics['confusion_matrix']}"
    )

    print(
        f"TEST_US_PER_SAMPLE="
        f"{result['test_microseconds_per_sample']:.6f}"
    )

    print(
        f"MODEL_SIZE_MIB="
        f"{result['model_size_bytes'] / 1024 / 1024:.3f}"
    )

    print(
        f"REPORT={report_path}"
    )

    print_memory(
        "FINAL"
    )

    print(
        "CICIDS_TREE_FINAL_RUN=PASS"
    )


if __name__ == "__main__":
    main()
