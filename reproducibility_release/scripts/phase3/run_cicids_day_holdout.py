import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow.dataset as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

DAY_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "cicids_day_holdout_protocol.json"
)

DAY_DISTRIBUTION_PATH = (
    ROOT
    / "artifacts"
    / "phase2"
    / "cicids_day_distribution.csv"
)

TREE_PROTOCOL_PATH = (
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
    / "day_holdout"
)

MODEL_DIRECTORY = (
    ROOT
    / "models"
    / "phase3"
    / "cicids"
    / "day_holdout"
)


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


def print_memory(label):
    values = {}

    for line in Path(
        "/proc/self/status"
    ).read_text().splitlines():
        if line.startswith("VmRSS:"):
            values["rss"] = (
                int(line.split()[1])
                / 1024
            )

        elif line.startswith("VmHWM:"):
            values["peak"] = (
                int(line.split()[1])
                / 1024
            )

    print(
        f"MEMORY_{label}_RSS_MIB="
        f"{values.get('rss', 0):.1f}"
    )

    print(
        f"MEMORY_{label}_PEAK_MIB="
        f"{values.get('peak', 0):.1f}"
    )


def load_features():
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

    if len(set(features)) != 78:
        raise RuntimeError(
            "Feature names are not unique"
        )

    return features


def load_day_counts():
    import pandas as pd

    frame = pd.read_csv(
        DAY_DISTRIBUTION_PATH
    )

    counts = {}

    for day in sorted(
        frame["source_day"].unique()
    ):
        subset = frame[
            frame["source_day"] == day
        ]

        benign = int(
            subset.loc[
                subset["binary_label"] == 0,
                "count"
            ].sum()
        )

        attack = int(
            subset.loc[
                subset["binary_label"] == 1,
                "count"
            ].sum()
        )

        counts[day] = {
            "benign": benign,
            "attack": attack,
            "total": benign + attack
        }

    return counts


def expected_counts(
    day_counts,
    days
):
    benign = sum(
        day_counts[day]["benign"]
        for day in days
    )

    attack = sum(
        day_counts[day]["attack"]
        for day in days
    )

    return {
        "benign": benign,
        "attack": attack,
        "total": benign + attack
    }


def validate_finite(
    matrix,
    name
):
    batch = 100000

    for start in range(
        0,
        len(matrix),
        batch
    ):
        stop = min(
            start + batch,
            len(matrix)
        )

        if not np.isfinite(
            matrix[start:stop]
        ).all():
            raise RuntimeError(
                f"Non-finite values in {name}"
            )


def load_partition(
    dataset,
    features,
    days,
    name,
    expected
):
    print(
        f"LOADING_{name.upper()}"
    )

    print(
        f"{name.upper()}_DAYS="
        f"{','.join(days)}"
    )

    table = dataset.to_table(
        columns=(
            features
            + [
                "binary_label"
            ]
        ),
        filter=(
            ds.field(
                "source_day"
            ).isin(
                days
            )
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

    matrix = frame.to_numpy(
        dtype=np.float32,
        copy=True
    )

    del frame
    release_memory()

    benign = int(
        (
            labels == 0
        ).sum()
    )

    attack = int(
        (
            labels == 1
        ).sum()
    )

    if len(labels) != expected["total"]:
        raise RuntimeError(
            f"{name} row mismatch: "
            f"{len(labels)} != "
            f"{expected['total']}"
        )

    if benign != expected["benign"]:
        raise RuntimeError(
            f"{name} benign mismatch"
        )

    if attack != expected["attack"]:
        raise RuntimeError(
            f"{name} attack mismatch"
        )

    validate_finite(
        matrix,
        name
    )

    print(
        f"{name.upper()}_ROWS="
        f"{len(labels)}"
    )

    print(
        f"{name.upper()}_BENIGN="
        f"{benign}"
    )

    print(
        f"{name.upper()}_ATTACK="
        f"{attack}"
    )

    print_memory(
        f"AFTER_{name.upper()}_LOAD"
    )

    return (
        matrix,
        labels
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

    benign_support = (
        tn + fp
    )

    attack_support = (
        tp + fn
    )

    fpr = (
        fp / benign_support
        if benign_support > 0
        else None
    )

    fnr = (
        fn / attack_support
        if attack_support > 0
        else None
    )

    result = {
        "samples": int(
            len(labels)
        ),
        "benign": int(
            benign_support
        ),
        "attack": int(
            attack_support
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "true_positives": tp,
        "false_positive_rate": (
            float(fpr)
            if fpr is not None
            else None
        ),
        "false_negative_rate": (
            float(fnr)
            if fnr is not None
            else None
        ),
        "specificity": (
            float(
                tn / benign_support
            )
            if benign_support > 0
            else None
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
        ],
        "binary_metrics_defined": (
            len(
                np.unique(
                    labels
                )
            )
            == 2
        )
    }

    if not result[
        "binary_metrics_defined"
    ]:
        result.update(
            {
                "balanced_accuracy": None,
                "precision_attack": None,
                "recall_attack": None,
                "f1_attack": None,
                "macro_f1": None,
                "weighted_f1": None,
                "roc_auc": None,
                "average_precision": None
            }
        )

        return result

    (
        precision,
        recall,
        f1,
        _
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[
            0,
            1
        ],
        zero_division=0
    )

    benign_recall = (
        tn
        / benign_support
    )

    attack_recall = (
        tp
        / attack_support
    )

    result.update(
        {
            "balanced_accuracy": float(
                (
                    benign_recall
                    + attack_recall
                )
                / 2
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
            )
        }
    )

    return result


def evaluate_rf(
    model,
    dataset,
    features,
    days,
    name,
    expected,
    threshold
):
    (
        matrix,
        labels
    ) = load_partition(
        dataset,
        features,
        days,
        name,
        expected
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

    samples = len(
        labels
    )

    del matrix
    del labels
    del probabilities

    release_memory()

    return {
        "prediction_seconds": float(
            elapsed
        ),
        "microseconds_per_sample": float(
            elapsed
            / samples
            * 1000000
        ),
        "metrics": metrics
    }


def run_rf(
    tree_protocol,
    dataset,
    features,
    partition,
    counts,
    seed,
    threads,
    threshold,
    run
):
    config = tree_protocol[
        "random_forest"
    ]

    (
        train_matrix,
        train_labels
    ) = load_partition(
        dataset,
        features,
        partition[
            "training_days"
        ],
        "train",
        counts[
            "train"
        ]
    )

    model = RandomForestClassifier(
        n_estimators=config[
            "n_estimators"
        ],
        criterion=config[
            "criterion"
        ],
        max_depth=config[
            "max_depth"
        ],
        min_samples_leaf=config[
            "min_samples_leaf"
        ],
        max_features=config[
            "max_features"
        ],
        max_samples=config[
            "max_samples"
        ],
        bootstrap=config[
            "bootstrap"
        ],
        class_weight=config[
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

    validation = evaluate_rf(
        model,
        dataset,
        features,
        partition[
            "validation_days"
        ],
        "validation",
        counts[
            "validation"
        ],
        threshold
    )

    test = evaluate_rf(
        model,
        dataset,
        features,
        partition[
            "test_days"
        ],
        "test",
        counts[
            "test"
        ],
        threshold
    )

    model_path = (
        MODEL_DIRECTORY
        / (
            f"rf_run_{run}_"
            f"seed_{seed}.joblib"
        )
    )

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    return {
        "training_seconds": float(
            training_seconds
        ),
        "best_iteration": None,
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


def run_xgb(
    tree_protocol,
    dataset,
    features,
    partition,
    counts,
    seed,
    threads,
    threshold,
    run
):
    config = tree_protocol[
        "xgboost"
    ]

    (
        train_matrix,
        train_labels
    ) = load_partition(
        dataset,
        features,
        partition[
            "training_days"
        ],
        "train",
        counts[
            "train"
        ]
    )

    negatives = int(
        (
            train_labels == 0
        ).sum()
    )

    positives = int(
        (
            train_labels == 1
        ).sum()
    )

    scale_pos_weight = (
        negatives
        / positives
    )

    train_dmatrix = (
        xgb.QuantileDMatrix(
            train_matrix,
            label=train_labels,
            feature_names=features,
            max_bin=config[
                "max_bin"
            ]
        )
    )

    del train_matrix
    del train_labels
    release_memory()

    (
        validation_matrix,
        validation_labels
    ) = load_partition(
        dataset,
        features,
        partition[
            "validation_days"
        ],
        "validation",
        counts[
            "validation"
        ]
    )

    validation_dmatrix = (
        xgb.QuantileDMatrix(
            validation_matrix,
            label=validation_labels,
            feature_names=features,
            max_bin=config[
                "max_bin"
            ],
            ref=train_dmatrix
        )
    )

    del validation_matrix
    release_memory()

    parameters = {
        "objective": config[
            "objective"
        ],
        "eval_metric": config[
            "eval_metric"
        ],
        "tree_method": config[
            "tree_method"
        ],
        "max_bin": config[
            "max_bin"
        ],
        "max_depth": config[
            "max_depth"
        ],
        "eta": config[
            "eta"
        ],
        "subsample": config[
            "subsample"
        ],
        "colsample_bytree": config[
            "colsample_bytree"
        ],
        "min_child_weight": config[
            "min_child_weight"
        ],
        "lambda": config[
            "lambda"
        ],
        "alpha": config[
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

    print(
        "TRAINING_BEGIN"
    )

    start = time.perf_counter()

    model = xgb.train(
        parameters,
        train_dmatrix,
        num_boost_round=config[
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
        early_stopping_rounds=config[
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

    validation_metrics = (
        metric_report(
            validation_labels,
            validation_probabilities,
            threshold
        )
    )

    validation_samples = len(
        validation_labels
    )

    del validation_probabilities
    del validation_labels
    del validation_dmatrix
    del train_dmatrix

    release_memory()

    (
        test_matrix,
        test_labels
    ) = load_partition(
        dataset,
        features,
        partition[
            "test_days"
        ],
        "test",
        counts[
            "test"
        ]
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

    test_samples = len(
        test_labels
    )

    del test_dmatrix
    del test_labels
    del test_probabilities

    release_memory()

    model_path = (
        MODEL_DIRECTORY
        / (
            f"xgb_run_{run}_"
            f"seed_{seed}.ubj"
        )
    )

    model.save_model(
        model_path
    )

    return {
        "training_seconds": float(
            training_seconds
        ),
        "best_iteration": (
            best_iteration
        ),
        "scale_pos_weight": float(
            scale_pos_weight
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
            "prediction_seconds": float(
                validation_seconds
            ),
            "microseconds_per_sample": float(
                validation_seconds
                / validation_samples
                * 1000000
            ),
            "metrics": validation_metrics
        },
        "test": {
            "prediction_seconds": float(
                test_seconds
            ),
            "microseconds_per_sample": float(
                test_seconds
                / test_samples
                * 1000000
            ),
            "metrics": test_metrics
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

    day_protocol = json.loads(
        DAY_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    tree_protocol = json.loads(
        TREE_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if arguments.seed not in tree_protocol[
        "seeds"
    ]:
        raise RuntimeError(
            "Seed not in frozen protocol"
        )

    if arguments.threads != tree_protocol[
        "threads"
    ]:
        raise RuntimeError(
            "Thread count differs from "
            "frozen tree protocol"
        )

    partition = day_protocol[
        "runs"
    ][
        arguments.run
    ]

    if partition[
        "run"
    ] != arguments.run:
        raise RuntimeError(
            "Day protocol run mismatch"
        )

    day_counts = load_day_counts()

    counts = {
        "train": expected_counts(
            day_counts,
            partition[
                "training_days"
            ]
        ),
        "validation": expected_counts(
            day_counts,
            partition[
                "validation_days"
            ]
        ),
        "test": expected_counts(
            day_counts,
            partition[
                "test_days"
            ]
        )
    }

    features = load_features()

    dataset = ds.dataset(
        str(
            DATASET_PATH
        ),
        format="parquet"
    )

    threshold = float(
        tree_protocol[
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

    print(
        f"MODEL={arguments.model}"
    )

    print(
        f"RUN={arguments.run}"
    )

    print(
        f"SEED={arguments.seed}"
    )

    print(
        f"TRAIN_DAYS="
        f"{','.join(partition['training_days'])}"
    )

    print(
        f"VALIDATION_DAYS="
        f"{','.join(partition['validation_days'])}"
    )

    print(
        f"TEST_DAYS="
        f"{','.join(partition['test_days'])}"
    )

    print(
        f"EVALUATION_SCOPE="
        f"{partition['evaluation_scope']}"
    )

    print(
        f"FEATURES={len(features)}"
    )

    print_memory(
        "START"
    )

    if arguments.model == "rf":
        result = run_rf(
            tree_protocol,
            dataset,
            features,
            partition,
            counts,
            arguments.seed,
            arguments.threads,
            threshold,
            arguments.run
        )
    else:
        result = run_xgb(
            tree_protocol,
            dataset,
            features,
            partition,
            counts,
            arguments.seed,
            arguments.threads,
            threshold,
            arguments.run
        )

    report = {
        "experiment": (
            "cicids2017_day_holdout"
        ),
        "model": arguments.model,
        "run": arguments.run,
        "seed": arguments.seed,
        "threshold": threshold,
        "evaluation_scope": partition[
            "evaluation_scope"
        ],
        "training_days": partition[
            "training_days"
        ],
        "validation_days": partition[
            "validation_days"
        ],
        "test_days": partition[
            "test_days"
        ],
        "expected_counts": counts,
        **result
    }

    report_path = (
        ARTIFACT_DIRECTORY
        / (
            f"{arguments.model}_"
            f"run_{arguments.run}_"
            f"seed_{arguments.seed}_"
            f"report.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
        + "\n",
        encoding="utf-8"
    )

    metrics = result[
        "test"
    ][
        "metrics"
    ]

    print()
    print(
        f"MODEL_NAME={arguments.model}"
    )

    print(
        f"RUN={arguments.run}"
    )

    print(
        f"TEST_DAY="
        f"{partition['test_days'][0]}"
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
        f"TEST_SAMPLES="
        f"{metrics['samples']}"
    )

    print(
        f"TEST_BENIGN="
        f"{metrics['benign']}"
    )

    print(
        f"TEST_ATTACK="
        f"{metrics['attack']}"
    )

    print(
        f"TEST_ACCURACY="
        f"{metrics['accuracy']:.8f}"
    )

    print(
        f"TEST_FALSE_POSITIVES="
        f"{metrics['false_positives']}"
    )

    print(
        f"TEST_FALSE_NEGATIVES="
        f"{metrics['false_negatives']}"
    )

    print(
        f"TEST_FPR="
        f"{metrics['false_positive_rate']:.8f}"
    )

    if metrics[
        "binary_metrics_defined"
    ]:
        print(
            f"TEST_BALANCED_ACCURACY="
            f"{metrics['balanced_accuracy']:.8f}"
        )

        print(
            f"TEST_MACRO_F1="
            f"{metrics['macro_f1']:.8f}"
        )

        print(
            f"TEST_ATTACK_F1="
            f"{metrics['f1_attack']:.8f}"
        )

        print(
            f"TEST_ATTACK_RECALL="
            f"{metrics['recall_attack']:.8f}"
        )

        print(
            f"TEST_FNR="
            f"{metrics['false_negative_rate']:.8f}"
        )

        print(
            f"TEST_ROC_AUC="
            f"{metrics['roc_auc']:.8f}"
        )
    else:
        print(
            "TEST_BINARY_METRICS_DEFINED=FALSE"
        )

    print(
        f"TEST_CONFUSION_MATRIX="
        f"{metrics['confusion_matrix']}"
    )

    print(
        f"TEST_US_PER_SAMPLE="
        f"{result['test']['microseconds_per_sample']:.6f}"
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
        "CICIDS_DAY_HOLDOUT_RUN=PASS"
    )


if __name__ == "__main__":
    main()
