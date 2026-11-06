"""Audit frozen USTC experiments at the original and first-five packet horizons.

Run: python3 scripts/paper/audit_comment3_4_final.py --root .
Automatically uses the existing Python 3.12 .venv-ml. Does not train models,
change partitions, select supports, generate traffic, or launch a capture.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
import warnings


def select_environment():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("root", nargs="?")
    parser.add_argument("--root", dest="root_option")
    args, _ = parser.parse_known_args()
    root = Path(args.root_option or args.root or Path(__file__).resolve().parents[2]).resolve()
    interpreter = root / ".venv-ml/bin/python"
    if interpreter.is_file() and Path(sys.prefix).resolve() != (root / ".venv-ml").resolve():
        os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    select_environment()

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found")


CLOSED_METRICS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "roc_auc_ovr_macro")
OPEN_METRICS = (
    "unknown_detection_auroc", "unknown_detection_average_precision",
    "known_false_rejection_rate", "unknown_recall", "open_set_macro_f1",
    "known_family_macro_f1_without_rejection",
)
STATUS_NAMES = (
    "CLOSED_ORIGINAL_REPRODUCTION", "CLOSED_N5_EVALUATION",
    "OPEN_ORIGINAL_REPRODUCTION", "OPEN_N5_EVALUATION", "DEPLOYMENT_RUN0_N5",
    "FIVE_PACKET_THRESHOLD_COMPARISON", "FEWSHOT_PER_FAMILY", "FEWSHOT_SUPPORT_NESTING",
    "FEATURE_CODE_CONSISTENCY", "CONTROLLER_PCAP_IAT_AGREEMENT", "ORIGINAL_ARTIFACTS_UNCHANGED",
)
                                                                           
                                                                              
PROBABILITY_ATOL = 2e-4
METRIC_ATOL = 1e-4
PREDICTION_MISMATCH_RATE = 1e-4


def average_precision_interval(known_scores, target_scores):
    """Bounds from moving opposing labels by one float32 epsilon.

    AP is discontinuous when opposite labels have almost tied scores. The
    bounds diagnose this independently of any saved metric; they do not alter
    model scores or the strict AUROC and classification reproduction gates.
    """
    labels = np.r_[np.zeros(len(known_scores), dtype=int), np.ones(len(target_scores), dtype=int)]
    scores = np.r_[known_scores, target_scores].astype(np.float64)
    epsilon = float(np.finfo(np.float32).eps)
    shift = np.where(labels == 1, epsilon, -epsilon)
    return (float(average_precision_score(labels, scores - shift)),
            float(average_precision_score(labels, scores + shift)))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_open(metrics):
    return {**{name: metrics[name] for name in OPEN_METRICS if name != "open_set_macro_f1"},
            "open_set_macro_f1": metrics["open_set"]["macro_f1"]}


def mapping(names):
    return dict(zip(names, range(len(names))))


class Audit:
    def __init__(self, args):
        self.root = Path(args.root_option or args.root or ".").resolve()
        self.out = self.root / "artifacts/paper/mentor_comment3_4_final"
        self.out.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.status = dict.fromkeys(STATUS_NAMES, "NOT_VERIFIED")
        self.errors, self.used, self.details, self.rows = [], {}, {}, {}
        self.log_path = self.out / "audit_log.txt"
        self.log_path.write_text("", encoding="utf-8")
        self.log(f"COMMAND=python3 scripts/paper/audit_comment3_4_final.py --root {args.root_option or args.root or '.'}")
        self.log(f"PROJECT_ROOT={self.root} PYTHON={sys.version.split()[0]} EXECUTABLE={sys.executable}")
        self.log(f"REPRODUCTION_TOLERANCES metric_abs={METRIC_ATOL:g} probability_abs={PROBABILITY_ATOL:g} prediction_mismatch_rate={PREDICTION_MISMATCH_RATE:g}; float32 backend boundary allowance; expected manuscript values are not outputs")
        self.before = self.frozen_snapshot()
        self.log(f"ORIGINAL_ARTIFACT_SNAPSHOT files={len(self.before)}")
        self.device = "not selected"
        self.frozen_expected = {}
        for relative in ("artifacts/phase3/ustc/closed_set/model_ready_sha256.txt",
                         "artifacts/phase3/ustc/closed_set/closed_set_sha256.txt",
                         "artifacts/phase3/ustc/open_set/open_set_sha256.txt"):
            path = self.path(relative)
            if path.is_file():
                for line in path.read_text().splitlines():
                    parts = line.split(None, 1)
                    if len(parts) == 2 and len(parts[0]) == 64:
                        name = parts[1].lstrip("*")
                        if name in self.frozen_expected and self.frozen_expected[name] != parts[0]:
                            raise ValueError(f"Conflicting frozen manifests for {name}")
                        self.frozen_expected[name] = parts[0]

    def configure(self):
        args = self.args
        sys.path.insert(0, str(self.root / "scripts/phase3"))
        import run_ustc_closed_set
        import run_ustc_open_set
        self.closed, self.opened = run_ustc_closed_set, run_ustc_open_set
        self.service = module_from_file("audit_inference_service", self.root / "scripts/phase4/run_inference_service.py")
        for relative in ("scripts/phase2/freeze_ustc_protocol.py", "scripts/phase3/build_ustc_closed_model_ready.py"):
            self.use(self.path(relative))
        self.closed_protocol = self.json(self.rebase(self.closed.PROTOCOL_PATH))
        self.open_protocol = self.json(self.rebase(self.opened.PROTOCOL_PATH))
        self.phase2 = self.json(self.path(self.open_protocol["rotation_source"]))
        self.deployment = self.json(self.root / "artifacts/phase4/phase4_protocol.json")
        self.families = self.closed_protocol["families"]
        self.rotations = self.phase2["open_set"]["runs_detail"]
        if len(self.families) != 10 or len(set(self.families)) != 10 or len(self.rotations) != 10:
            raise ValueError("Expected ten distinct families and ten frozen rotations")
        self.closed.set_seed(int(self.open_protocol["seed"]))
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu or auto")
        self.device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
        self.log(f"INFERENCE_DEVICE={self.device} THREADS={torch.get_num_threads()} TF32=False")
        self.norm_cache = {}

    def log(self, message):
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Artifact path outside --root: {relative}")
        return path

    def rebase(self, path):
        return self.path(path.relative_to(self.closed.ROOT))

    def use(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact missing: {path}")
        relative = str(path.relative_to(self.root))
        self.used[relative] = self.before.get(relative) or sha256(path)
        expected = self.frozen_expected.get(relative)
        if expected is not None and self.used[relative] != expected:
            raise ValueError(f"Frozen manifest mismatch before use: {path}; expected={expected}; observed={self.used[relative]}")
        self.log(f"ARTIFACT={relative} SHA256={self.used[relative]}")
        return path

    def json(self, path):
        return read_json(self.use(path))

    def checkpoint(self, path):
                                                                                
        value = torch.load(self.use(path), map_location="cpu", weights_only=False)
        if not isinstance(value, dict) or not isinstance(value.get("state_dict"), dict):
            raise ValueError(f"{path}: expected project checkpoint with state_dict; actual type={type(value)}")
        self.log(f"CHECKPOINT_SCHEMA={path.name} keys={list(value)}")
        return value

    def frozen_snapshot(self):
        paths = set()
        for directory in ("artifacts/phase2", "artifacts/phase3", "artifacts/phase4", "models"):
            paths.update(p for p in (self.root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        paths.update(p for p in (self.root / "artifacts/paper").rglob("*") if p.is_file() and not p.is_relative_to(self.out))
        for directory in ("scripts/phase2", "scripts/phase3", "scripts/phase4", "controller"):
            paths.update((self.root / directory).rglob("*.py"))
        return {str(p.relative_to(self.root)): sha256(p) for p in sorted(paths)}

    def verify_manifests(self):
        expected = {}
        for relative in ("artifacts/phase3/ustc/closed_set/model_ready_sha256.txt",
                         "artifacts/phase3/ustc/closed_set/closed_set_sha256.txt",
                         "artifacts/phase3/ustc/open_set/open_set_sha256.txt"):
            path = self.use(self.path(relative))
            for line in path.read_text().splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and len(parts[0]) == 64:
                    name = parts[1].lstrip("*")
                    if name in self.used:
                        if name in expected and expected[name] != parts[0]:
                            raise ValueError(f"Conflicting frozen hashes for {name}: {path}")
                        expected[name] = parts[0]
        mismatches = [name for name, digest in expected.items() if self.used[name] != digest]
        if mismatches:
            raise ValueError(f"Frozen manifest SHA256 mismatch: {mismatches}")
        self.details["frozen_manifest_verification"] = {"status": "PASS", "checked_files": len(expected)}

    def load_data(self):
        path = self.use(self.path(self.open_protocol["model_ready_source"]))
        if path != self.rebase(self.closed.DATA_PATH):
            raise ValueError(f"Closed/open prepared dataset paths disagree: {path}, {self.closed.DATA_PATH}")
        required = ["class_name", "flow_id", "sequence_packets", "signed_packet_sizes",
                    "interarrival_seconds", "closed_split", "adaptation_partition", "representation_fingerprint"]
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        self.log(f"PARQUET_SCHEMA path={path} rows={parquet.metadata.num_rows} schema={parquet.schema_arrow}")
        if not set(required).issubset(names):
            raise ValueError(f"{path}: actual columns={names}; required={required}; missing={sorted(set(required)-set(names))}")
        frame = pd.concat((b.to_pandas() for b in parquet.iter_batches(columns=required, batch_size=8192)), ignore_index=True)
        counts = {k: int(v) for k, v in frame.closed_split.value_counts().items()}
        expected = {k: self.closed_protocol["split"][k] for k in ("train", "validation", "test")}
        if counts != expected or len(frame) != sum(expected.values()):
            raise ValueError(f"{path}: actual split counts={counts}; expected frozen protocol={expected}; columns={names}")
        if set(frame.class_name) != set(self.families) or frame.flow_id.duplicated().any():
            raise ValueError(f"{path}: unexpected families or duplicate flow_id")
        for row in frame.itertuples(index=False):
            sizes, iats, n = row.signed_packet_sizes, row.interarrival_seconds, row.sequence_packets
            if not 3 <= n <= 50 or len(sizes) != n or len(iats) != n:
                raise ValueError(f"{path}: flow={row.flow_id} sequence_packets={n}; array lengths sizes={len(sizes)}, iats={len(iats)}; expected equal lengths 3..50")
            if not np.isfinite(sizes).all() or not np.isfinite(iats).all() or np.any(iats < 0) or iats[0] != 0:
                raise ValueError(f"{path}: flow={row.flow_id} invalid raw features; IATs must be nonnegative seconds with first IAT zero")
        manifest_path = self.use(self.path(self.closed_protocol["split"]["source"]))
        metadata = [c for c in required if c not in ("signed_packet_sizes", "interarrival_seconds")]
        manifest = pd.read_parquet(manifest_path, columns=metadata)
        a = frame.set_index(["class_name", "flow_id"])
        b = manifest.set_index(["class_name", "flow_id"])
        if a.index.duplicated().any() or b.index.duplicated().any() or set(a.index) != set(b.index):
            raise ValueError(f"Prepared/partition manifest identity mismatch: {path}, {manifest_path}")
        aligned = a.loc[b.index]
        for column in b.columns:
            if not np.array_equal(aligned[column].to_numpy(), b[column].to_numpy()):
                raise ValueError(f"Prepared/partition manifest mismatch column={column}: {path}, {manifest_path}")
        for partition in ("closed_split", "adaptation_partition"):
            if (frame.groupby(["class_name", "representation_fingerprint"])[partition].nunique() > 1).any():
                raise ValueError(f"Representation leakage across {partition}: {path}")
        self.frame = frame
        self.details["dataset"] = {"path": str(path.relative_to(self.root)), "rows": len(frame), "columns": names, "split_counts": counts,
                                   "closed_test_n5_eligible": int(((frame.closed_split == "test") & (frame.sequence_packets >= 5)).sum())}
        self.log(f"DATASET_POPULATED=PASS rows={len(frame)} split_counts={counts} closed_test_n5_eligible={self.details['dataset']['closed_test_n5_eligible']}")
        self.log(f"PARQUET_SAMPLE={json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in frame.iloc[0].to_dict().items()})}")

    def verify_normalization(self, checkpoint, report, known):
        key = tuple(known)
        if key not in self.norm_cache:
            train = self.frame[self.frame.class_name.isin(known) & (self.frame.closed_split == "train")]
            self.norm_cache[key] = self.opened.fit_normalization(train)
            self.log(f"NORMALIZATION_SOURCE=known_train families={list(known)} rows={len(train)} parameters={json.dumps(self.norm_cache[key], sort_keys=True)}")
        for name, value in self.norm_cache[key].items():
            if not np.isclose(checkpoint["normalization"][name], value, atol=1e-12, rtol=1e-12):
                raise ValueError(f"Training-derived normalization mismatch {name}: saved={checkpoint['normalization'][name]}, derived={value}")
        if checkpoint["normalization"] != report["normalization"]:
            raise ValueError("Checkpoint/report normalization mismatch")

    def error(self, task, exc):
        detail = {"task": task, "error": f"{type(exc).__name__}: {exc}"}
        self.errors.append(detail)
        self.log(f"ERROR task={task} {detail['error']}\n{traceback.format_exc()}")

    def save(self, name, rows):
        self.rows[name] = rows
        write_csv(self.out / name, rows)

    def aggregate(self, rows, metrics, variability):
        result = []
        groups = sorted(set((r.get("method", "cnn_bilstm"), r["horizon"], r["cohort"]) for r in rows))
        for method, horizon, cohort in groups:
            group = [r for r in rows if (r.get("method", "cnn_bilstm"), r["horizon"], r["cohort"]) == (method, horizon, cohort)]
            for metric in metrics:
                values = [r[metric] for r in group]
                result.append({"method": method, "horizon": horizon, "cohort": cohort, "metric": metric,
                               "n": len(values), "mean": statistics.mean(values), "sample_sd": sample_sd(values),
                               "variability": variability, "verified": all(r["verified"] for r in group)})
        return result

    def print_aggregate(self, name, summary):
        items = [r for r in summary if r["horizon"] == 5 and r["verified"]]
        for method in sorted(set(r["method"] for r in items)):
            group = [r for r in items if r["method"] == method]
            text = " ".join(f"{r['metric']}={r['mean']:.6f}±{r['sample_sd']:.6f}" if r["sample_sd"] is not None
                            else f"{r['metric']}={r['mean']:.6f} sample_sd=not_computable" for r in group)
            text = f"runs={group[0]['n']} " + text
            self.log(f"{name} method={method} {text}")

    def closed_audit(self):
        frame = self.frame[self.frame.closed_split == "test"].reset_index(drop=True)
        eligible = (frame.sequence_packets >= 5).to_numpy()
        classes = self.families
        rows, reproduction, family_rows, confusion_rows = [], [], [], []
        for seed in self.closed_protocol["seeds"]:
            report_path = self.rebase(self.closed.ARTIFACT_DIRECTORY) / f"cnn_bilstm_seed_{seed}_report.json"
            try:
                report = self.json(report_path)
                ck_path = self.path(report["model_path"])
                ck = self.checkpoint(ck_path)
                if ck["seed"] != seed or ck["class_names"] != classes or report["families"] != classes:
                    raise ValueError(f"{ck_path}: seed/class ordering mismatch")
                self.verify_normalization(ck, report, classes)
                proto = self.closed_protocol["model"]
                model = self.closed.CNNBiLSTM(len(classes), proto["bilstm_hidden_size_per_direction"], proto["dropout"]).to(self.device)
                model.load_state_dict(ck["state_dict"], strict=True)
                model.eval()
                x, lengths, labels = self.closed.encode_frame(frame, ck["normalization"], mapping(classes))
                                                                              
                                                                          
                _, _, p50 = self.opened.predict_ce(model, x, lengths, self.args.batch_size, self.device)
                baseline, prediction = self.closed.calculate_metrics(labels, p50, classes)
                ref_path = self.use(self.path(report["test_predictions_path"]))
                reference = pd.read_parquet(ref_path)
                if reference.flow_id.duplicated().any() or set(reference.flow_id) != set(frame.flow_id):
                    raise ValueError(f"{ref_path}: frozen test identity mismatch")
                reference = reference.set_index("flow_id").loc[frame.flow_id]
                if not np.array_equal(reference.true_label.to_numpy(), labels):
                    raise ValueError(f"{ref_path}: saved labels disagree with checkpoint class order")
                saved_p = reference[[f"probability_{c}" for c in classes]].to_numpy(dtype=np.float32)
                difference = np.abs(saved_p - p50)
                mismatches = prediction != reference.predicted_label.to_numpy()
                top = np.sort(saved_p, axis=1)[:, -2:]
                near_boundary = top[:, 1] - top[:, 0] <= 2 * PROBABILITY_ATOL
                metric_diffs = {m: abs(baseline[m] - report["test_metrics"][m]) for m in CLOSED_METRICS}
                passed = (float(difference.max()) <= PROBABILITY_ATOL and float(mismatches.mean()) <= PREDICTION_MISMATCH_RATE
                          and np.all(~mismatches | near_boundary) and max(metric_diffs.values()) <= METRIC_ATOL)
                reproduction.append({"seed": seed, "status": "PASS" if passed else "FAIL", "test_rows": len(frame),
                                     "probability_max_abs_error": float(difference.max()), "probability_mean_abs_error": float(difference.mean()),
                                     "prediction_mismatches": int(mismatches.sum()), "metric_max_abs_error": max(metric_diffs.values()),
                                     **{f"recreated_{m}": baseline[m] for m in CLOSED_METRICS},
                                     **{f"saved_{m}": report["test_metrics"][m] for m in CLOSED_METRICS}})
                self.log(f"CLOSED_REPRODUCTION seed={seed} status={'PASS' if passed else 'FAIL'} probability_max_abs={difference.max():.8g} mismatches={mismatches.sum()} metric_max_abs={max(metric_diffs.values()):.8g}")
                if not passed:
                    raise ValueError(f"{ck_path}: original full-test reproduction failed; differences={metric_diffs}; new metrics withheld")
                x5 = x[eligible].copy()
                x5[:, 5:, :] = 0
                n5 = np.minimum(lengths[eligible], 5)
                _, _, p5 = self.opened.predict_ce(model, x5, n5, self.args.batch_size, self.device)
                for horizon, cohort, y, probabilities in (
                    (50, "original_full_test", labels, p50),
                    (50, "matched_ge5_test", labels[eligible], p50[eligible]),
                    (5, "matched_ge5_test", labels[eligible], p5),
                ):
                    metrics, _ = self.closed.calculate_metrics(y, probabilities, classes)
                    rows.append({"seed": seed, "horizon": horizon, "cohort": cohort, "samples": len(y), "verified": True,
                                 "checkpoint": str(ck_path.relative_to(self.root)), **{m: metrics[m] for m in CLOSED_METRICS}})
                    for name, values in metrics["per_class"].items():
                        family_rows.append({"seed": seed, "horizon": horizon, "cohort": cohort, "family": name, **values})
                    for i, actual in enumerate(classes):
                        for j, predicted in enumerate(classes):
                            confusion_rows.append({"seed": seed, "horizon": horizon, "cohort": cohort, "true_family": actual,
                                                   "predicted_family": predicted, "count": metrics["confusion_matrix"][i][j]})
                self.log(f"CLOSED_N5 seed={seed} eligible={eligible.sum()} accuracy={rows[-1]['accuracy']:.6f} macro_f1={rows[-1]['macro_f1']:.6f} matched50_macro_f1={rows[-2]['macro_f1']:.6f}")
            except Exception as exc:
                self.error(f"closed seed={seed} report={report_path}", exc)
        complete = len(reproduction) == 5 and all(r["status"] == "PASS" for r in reproduction)
        self.status["CLOSED_ORIGINAL_REPRODUCTION"] = "PASS" if complete else "FAIL"
        self.status["CLOSED_N5_EVALUATION"] = "PASS" if complete and len(rows) == 15 else "FAIL"
        summary = self.aggregate(rows, CLOSED_METRICS, "stochastic seed variability on one frozen split")
        for filename, data in (("closed_original_reproduction.csv", reproduction), ("closed_five_packet_per_seed.csv", rows),
                               ("closed_five_packet_summary.csv", summary), ("closed_five_packet_per_family.csv", family_rows),
                               ("closed_five_packet_confusion_matrix.csv", confusion_rows)):
            self.save(filename, data)
        self.print_aggregate("CLOSED_N5_MEAN", summary)

    def partition(self, names, column, value):
        return self.frame[self.frame.class_name.isin(names) & (self.frame[column] == value)].reset_index(drop=True)

    def open_predict(self, model, ck, frame, horizon):
        if frame.empty:
            raise ValueError(f"Empty {horizon}-packet input cohort; columns={list(frame.columns)}")
        x, lengths = self.opened.encode_inputs(frame, ck["normalization"])
        if horizon == 5:
            if np.any(lengths < 5):
                raise ValueError("Five-packet cohort includes a flow with fewer than five valid tokens")
            x[:, 5:, :] = 0
            lengths[:] = 5
        if ck["method"] == "ce_msp":
            prediction, scores, _ = self.opened.predict_ce(model, x, lengths, self.args.batch_size, self.device)
        else:
            embeddings = self.opened.embed(model, x, lengths, self.args.batch_size, self.device)
            prediction, scores, _ = self.opened.prototype_predictions(embeddings, ck["known_prototypes"])
        return prediction, scores

    def open_metrics(self, names, known, kp, ks, tp, ts, threshold):
        labels = self.opened.labels_for(known, mapping(names))
        metrics = self.opened.zero_shot_metrics(labels, kp, ks, tp, ts, threshold, names)
        return flatten_open(metrics), metrics

    def open_audit(self):
        rows, reproduction, thresholds, per_class, confusion = [], [], [], [], []
        self.deployment_input = None
        for rotation in sorted(self.rotations, key=lambda r: r["run"]):
            run, names = rotation["run"], rotation["known_families"]
            target, calibration = rotation["target_unknown_family"], rotation["calibration_unknown_family"]
            if len(names) != 8 or len(set(names + [target, calibration])) != 10:
                raise ValueError(f"Invalid frozen rotation={rotation}")
            known = self.partition(names, "closed_split", "test")
            query = self.partition([target], "adaptation_partition", "query_pool")
            validation = self.partition(names, "closed_split", "validation")
            cal = self.partition([calibration], "adaptation_partition", "calibration_pool")
            known5, query5, val5, cal5 = (f[f.sequence_packets >= 5].reset_index(drop=True) for f in (known, query, validation, cal))
            for method in ("ce_msp", "supcon_proto"):
                report_path = self.rebase(self.opened.ARTIFACT_DIRECTORY) / f"{method}_run_{run}_report.json"
                try:
                    report = self.json(report_path)
                    ck_path = self.path(report["model_path"])
                    ck = self.checkpoint(ck_path)
                    for key, value in {"method": method, "run": run, "known_families": names,
                                       "target_unknown_family": target, "calibration_unknown_family": calibration}.items():
                        if ck[key] != value or report[key] != value:
                            raise ValueError(f"{ck_path}: {key} mapping disagrees with frozen rotation")
                    self.verify_normalization(ck, report, names)
                    counts = {"known_test": len(known), "known_validation": len(validation), "calibration_pool": len(cal),
                              "target_query_pool": len(query), "known_train": len(self.partition(names, "closed_split", "train")),
                              "target_support_pool": len(self.partition([target], "adaptation_partition", "support_pool"))}
                    if report["partition_counts"] != counts:
                        raise ValueError(f"{report_path}: actual partitions={counts}; expected={report['partition_counts']}")
                    hidden = self.open_protocol["encoder"]["bilstm_hidden_size_per_direction"]
                    model = self.opened.CEModel(hidden, len(names)) if method == "ce_msp" else self.opened.Encoder(hidden)
                    model.load_state_dict(ck["state_dict"], strict=True)
                    model = model.to(self.device).eval()
                    if method == "supcon_proto":
                        proto = ck["known_prototypes"]
                        if proto.shape != (8, hidden * 2) or not np.allclose(np.linalg.norm(proto, axis=1), 1, atol=1e-6):
                            raise ValueError(f"{ck_path}: invalid frozen prototypes shape={proto.shape}")
                    threshold = report["threshold_calibration"]["threshold"]
                    kp, ks = self.open_predict(model, ck, known, 50)
                    tp, ts = self.open_predict(model, ck, query, 50)
                    baseline, _ = self.open_metrics(names, known, kp, ks, tp, ts, threshold)
                    reference = flatten_open(report["zero_shot"])
                    differences = {m: abs(baseline[m] - reference[m]) for m in OPEN_METRICS}
                    ap_name = "unknown_detection_average_precision"
                    ap_low, ap_high = average_precision_interval(ks, ts)
                    ap_close = differences[ap_name] <= METRIC_ATOL
                    ap_compatible = ap_close or ap_low <= reference[ap_name] <= ap_high
                    passed = max(v for k, v in differences.items() if k != ap_name) <= METRIC_ATOL and ap_compatible
                    reproduction.append({"method": method, "run": run, "target_family": target, "status": "PASS" if passed else "FAIL",
                                         "max_metric_abs_error": max(differences.values()),
                                         "ap_verification": "within_scalar_tolerance" if ap_close else "float32_rank_interval" if ap_compatible else "FAIL",
                                         "ap_rank_interval_low": ap_low, "ap_rank_interval_high": ap_high,
                                         "ap_rank_score_epsilon": float(np.finfo(np.float32).eps),
                                         **{f"recreated_{m}": baseline[m] for m in OPEN_METRICS},
                                         **{f"saved_{m}": reference[m] for m in OPEN_METRICS}})
                    self.log(f"OPEN_REPRODUCTION method={method} run={run} target={target} status={'PASS' if passed else 'FAIL'} saved_auroc={reference['unknown_detection_auroc']:.8f} reproduced_auroc={baseline['unknown_detection_auroc']:.8f} max_metric_abs={max(differences.values()):.8g}")
                    if not ap_close:
                        self.log(f"OPEN_AP_NUMERICAL_CHECK method={method} run={run} saved={reference[ap_name]:.8f} reconstructed={baseline[ap_name]:.8f} one_float32_epsilon_interval=[{ap_low:.8f},{ap_high:.8f}] compatible={ap_compatible}; AUROC and classification gates remain {METRIC_ATOL:g}")
                    if not passed:
                        raise ValueError(f"{ck_path}: original per-rotation reproduction failed; differences={differences}; new metrics withheld")
                    km, qm = (known.sequence_packets >= 5).to_numpy(), (query.sequence_packets >= 5).to_numpy()
                    matched50, _ = self.open_metrics(names, known5, kp[km], ks[km], tp[qm], ts[qm], threshold)
                    kp5, ks5 = self.open_predict(model, ck, known5, 5)
                    tp5, ts5 = self.open_predict(model, ck, query5, 5)
                    matched5, detail5 = self.open_metrics(names, known5, kp5, ks5, tp5, ts5, threshold)
                    _, vs5 = self.open_predict(model, ck, val5, 5)
                    _, cs5 = self.open_predict(model, ck, cal5, 5)
                    new_threshold = self.opened.select_threshold(vs5, cs5)["threshold"]
                    recalibrated, _ = self.open_metrics(names, known5, kp5, ks5, tp5, ts5, new_threshold)
                    shared = {"method": method, "run": run, "target_family": target, "calibration_family": calibration,
                              "checkpoint": str(ck_path.relative_to(self.root)), "verified": True}
                    for horizon, cohort, n_known, n_target, metrics in (
                        (50, "original_full_test_and_query", len(known), len(query), baseline),
                        (50, "matched_ge5_test_and_query", len(known5), len(query5), matched50),
                        (5, "matched_ge5_test_and_query", len(known5), len(query5), matched5),
                    ):
                        rows.append({**shared, "horizon": horizon, "cohort": cohort, "n_known": n_known, "n_unknown": n_target,
                                     "threshold": threshold, "operating_point": "original_threshold", **metrics})
                    for point, cutoff, metrics in (("original_threshold", threshold, matched5), ("offline_n5_recalibrated", new_threshold, recalibrated)):
                        thresholds.append({**shared, "horizon": 5, "operating_point": point, "threshold": cutoff,
                                           "n_known": len(known5), "n_unknown": len(query5), "calibration_known_n": len(val5),
                                           "calibration_unknown_n": len(cal5), "target_used_in_calibration": False,
                                           "used_in_original_phase4": method == "supcon_proto" and run == 0 and point == "original_threshold", **metrics})
                    for family, values in detail5["open_set"]["per_class"].items():
                        per_class.append({**shared, "horizon": 5, "family": family, **values})
                    confusion.append({**shared, "horizon": 5, "class_order": json.dumps(names + ["unknown"]),
                                      "confusion_matrix": json.dumps(detail5["open_set"]["confusion_matrix"])})
                    if method == "supcon_proto" and run == 0:
                        self.deployment_input = (known5, query5, val5, cal5, report, ck_path)
                    self.log(f"OPEN_N5 method={method} run={run} eligible_known={len(known5)} eligible_unknown={len(query5)} " + " ".join(f"{m}={matched5[m]:.6f}" for m in OPEN_METRICS))
                    self.log(f"THRESHOLD_N5 method={method} run={run} original={threshold:.8g} recalibrated={new_threshold:.8g} calibration={calibration} target_excluded={target}")
                except Exception as exc:
                    self.error(f"open method={method} run={run} report={report_path}", exc)
        complete = len(reproduction) == 20 and all(r["status"] == "PASS" for r in reproduction)
        self.status["OPEN_ORIGINAL_REPRODUCTION"] = "PASS" if complete else "FAIL"
        self.status["OPEN_N5_EVALUATION"] = "PASS" if complete and len(rows) == 60 else "FAIL"
        self.status["FIVE_PACKET_THRESHOLD_COMPARISON"] = "PASS" if len(thresholds) == 40 else "FAIL"
        self.details["open_prediction_reference"] = "No saved per-flow open-set predictions/scores exist. AUROC and discrete metrics use the fixed scalar tolerance; AP uses that tolerance or the reported rank-sensitivity interval from one float32 epsilon. Actual reconstructed scores/metrics are never replaced by saved values. Partition identities, class ordering, train-only normalization and frozen prototypes are checked."
        summary = self.aggregate(rows, OPEN_METRICS, "between-target-family task variability; not stochastic seed variability")
        for filename, data in (("open_original_reproduction.csv", reproduction), ("open_five_packet_per_rotation.csv", rows),
                               ("open_five_packet_summary.csv", summary), ("five_packet_threshold_comparison.csv", thresholds),
                               ("open_five_packet_per_family.csv", per_class), ("open_five_packet_confusion_matrix.csv", confusion)):
            self.save(filename, data)
        self.print_aggregate("OPEN_N5_MEAN", summary)

    def deployment_predict(self, runtime, frame):
        predictions, scores = [], []
        for start in range(0, len(frame), self.args.batch_size):
            batch = frame.iloc[start:start + self.args.batch_size]
            inputs = []
            for row in batch.itertuples(index=False):
                encoded, _ = runtime._encode(row.signed_packet_sizes[:5], row.interarrival_seconds[:5])
                inputs.append(encoded)
            x = torch.cat(inputs)
            lengths = torch.full((len(x),), 5, dtype=torch.int64)
            with torch.inference_mode():
                embedding = F.normalize(runtime.encoder(x, lengths), p=2, dim=1)
                similarities = torch.clamp(embedding @ runtime.prototypes.T, -1, 1)
                maximum, prediction = similarities.max(dim=1)
            predictions.append(prediction.cpu().numpy())
            scores.append((1 - maximum).cpu().numpy())
        return np.concatenate(predictions), np.concatenate(scores)

    def deployment_audit(self):
        if self.deployment_input is None:
            raise ValueError("Deployed run-0 original reproduction has not passed; deployment metrics withheld")
        known, query, validation, calibration, report, checkpoint = self.deployment_input
        config = self.deployment["inference"]
        if checkpoint != self.path(config["model"]) or self.path(config["model"]) != self.rebase(self.service.MODEL_PATH):
            raise ValueError(f"Deployment model mapping disagrees: {config}, {checkpoint}, {self.service.MODEL_PATH}")
        if self.path(config["report"]) != self.rebase(self.service.REPORT_PATH) or report["target_unknown_family"] != "Cridex":
            raise ValueError("Deployment report or Cridex target mapping disagrees")
        for name in ("MODEL_PATH", "REPORT_PATH", "PROTOCOL_PATH", "FROZEN_MANIFEST_PATH"):
            setattr(self.service, name, self.rebase(getattr(self.service, name)))
        self.service.ROOT = self.root
        runtime = self.service.Runtime(str(self.device))
        self.use(checkpoint)
        model_hashes, observed_thresholds = [], []
        for path in sorted((self.root / "artifacts/phase4").rglob("inference_service.log")):
            for line in path.read_text().splitlines():
                if "MODEL_SHA256=" in line:
                    model_hashes.append(line.split("MODEL_SHA256=", 1)[1].split()[0])
        for path in sorted((self.root / "artifacts/phase4").rglob("inference_controller_metrics.jsonl")):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if "threshold" in record:
                    observed_thresholds.append(record["threshold"])
        if (not model_hashes or set(model_hashes) != {runtime.model_hash}
                or not observed_thresholds or set(observed_thresholds) != {runtime.threshold}):
            raise ValueError(f"Phase4 telemetry differs from runtime: model_hashes={set(model_hashes)}, thresholds={set(observed_thresholds)}")
        kp, ks = self.deployment_predict(runtime, known)
        tp, ts = self.deployment_predict(runtime, query)
        _, vs = self.deployment_predict(runtime, validation)
        _, cs = self.deployment_predict(runtime, calibration)
        cutoff = self.opened.select_threshold(vs, cs)["threshold"]
        checks = 0
        for frame, prediction, score in ((known, kp, ks), (query, tp, ts)):
            for i in np.linspace(0, len(frame) - 1, min(32, len(frame)), dtype=int):
                row = frame.iloc[i]
                actual = runtime.infer(row.signed_packet_sizes[:5], row.interarrival_seconds[:5])
                if (actual["nearest_known_family"] != runtime.known_families[prediction[i]]
                        or abs(actual["unknown_score"] - float(score[i])) > 2e-6):
                    raise ValueError(f"Service batch/single inference mismatch flow={row.flow_id}: {actual}, batch_score={score[i]}")
                checks += 1
        rows = []
        for point, threshold in (("original_deployed_threshold", runtime.threshold), ("offline_n5_recalibrated", cutoff)):
            metrics, detail = self.open_metrics(runtime.known_families, known, kp, ks, tp, ts, threshold)
            known_metrics = self.opened.classification_metrics(self.opened.labels_for(known, mapping(runtime.known_families)), kp, runtime.known_families)
            rows.append({"method": "supcon_proto", "run": 0, "target_family": "Cridex", "calibration_family": report["calibration_unknown_family"],
                         "horizon": 5, "n_known": len(known), "n_unknown": len(query), "operating_point": point,
                         "threshold": threshold, "known_accuracy_before_rejection": known_metrics["accuracy"],
                         "known_balanced_accuracy_before_rejection": known_metrics["balanced_accuracy"],
                         "cridex_detection_precision": detail["open_set"]["per_class"]["unknown"]["precision"],
                         "cridex_detection_f1": detail["open_set"]["per_class"]["unknown"]["f1"],
                         "checkpoint": str(checkpoint.relative_to(self.root)), "model_sha256": runtime.model_hash,
                         "verified": True, "used_in_original_phase4": point == "original_deployed_threshold", **metrics})
            self.log(f"DEPLOYMENT_RUN0_N5_METRICS point={point} threshold={threshold:.8g} " + " ".join(f"{k}={metrics[k]:.6f}" for k in OPEN_METRICS))
        self.save("deployment_run0_five_packet.csv", rows)
        self.details["deployment"] = {"checkpoint": config["model"], "report": config["report"], "model_sha256": runtime.model_hash,
                                      "selection": "fixed rotation index 0; not selected by accuracy", "service_single_flow_checks": checks,
                                      "measurement": "offline USTC first-five inputs with frozen Phase4 runtime; not SDN traffic accuracy"}
        self.status["DEPLOYMENT_RUN0_N5"] = "PASS"

    def fewshot_audit(self):
        support = pd.read_csv(self.use(self.path(self.open_protocol["support_source"])), dtype={"flow_id": str})
        wide, summaries, long_rows, nesting = [], [], [], []
        identity = self.frame.set_index("flow_id")
        for rotation in sorted(self.rotations, key=lambda r: self.families.index(r["target_unknown_family"])):
            run, target = rotation["run"], rotation["target_unknown_family"]
            path = self.rebase(self.opened.ARTIFACT_DIRECTORY) / f"supcon_proto_run_{run}_report.json"
            report = self.json(path)
            if report["target_unknown_family"] != target or report["known_families"] != rotation["known_families"]:
                raise ValueError(f"{path}: few-shot rotation mapping mismatch")
            row = {"target_family": target, "run": run, "source": str(path.relative_to(self.root))}
            prior = []
            for k in (1, 5, 10):
                block = report["few_shot"][str(k)]
                metrics, ids = block["nine_family_without_rejection"], block["support_flow_ids"]
                values = metrics["per_class"][target]
                frozen = support[(support.run == run) & (support.target_family == target) & (support.k == k)].sort_values("support_rank")
                actual = identity.loc[ids]
                nested = ids == frozen.flow_id.tolist() and len(set(ids)) == k and ids[:len(prior)] == prior
                nested = nested and frozen.support_rank.tolist() == list(range(1, k + 1))
                nested = nested and bool(((actual.class_name == target) & (actual.adaptation_partition == "support_pool")).all())
                nesting.append({"run": run, "target_family": target, "k": k, "status": "PASS" if nested else "FAIL", "support_flow_ids": json.dumps(ids)})
                prior = ids
                order = list(rotation["known_families"]) + [target]
                matrix = np.asarray(metrics["confusion_matrix"])
                if matrix.shape != (9, 9) or np.any(matrix < 0):
                    raise ValueError(f"{path}: invalid nine-family confusion shape={matrix.shape}")
                diagonals = matrix.diagonal()
                denominators = matrix.sum(axis=0) + matrix.sum(axis=1)
                class_f1 = np.divide(2 * diagonals, denominators, out=np.zeros(9, dtype=float), where=denominators != 0)
                if abs(float(class_f1.mean()) - metrics["macro_f1"]) > 1e-12:
                    raise ValueError(f"{path}: K={k} nine-family macro-F1 disagrees with saved confusion")
                index = order.index(target)
                tp = matrix[index, index]
                precision = tp / matrix[:, index].sum() if matrix[:, index].sum() else 0.0
                recall = tp / matrix[index].sum() if matrix[index].sum() else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                if any(abs(a - b) > 1e-12 for a, b in ((precision, values["precision"]), (recall, values["recall"]), (f1, values["f1"]))):
                    raise ValueError(f"{path}: K={k} target PR/F1 disagrees with saved confusion")
                for metric in ("precision", "recall", "f1"):
                    row[f"target_{metric}_K{k}"] = values[metric]
                row[f"nine_family_macro_f1_K{k}"] = metrics["macro_f1"]
                long_rows.append({"k": k, "target_family": target, "target_precision": precision, "target_recall": recall,
                                  "target_f1": f1, "nine_family_macro_f1": metrics["macro_f1"]})
            wide.append(row)
        original = pd.read_csv(self.use(self.rebase(self.opened.ARTIFACT_DIRECTORY) / "few_shot_summary.csv")).set_index("k")
        rounded = {1: (0.296120, 0.263053), 5: (0.439168, 0.301118), 10: (0.422812, 0.312041)}
        for k in (1, 5, 10):
            group = [r for r in long_rows if r["k"] == k]
            for metric in ("target_precision", "target_recall", "target_f1", "nine_family_macro_f1"):
                values = [r[metric] for r in group]
                mean, sd = statistics.mean(values), sample_sd(values)
                expected_mean, expected_sd = original.loc[k, f"{metric}_mean"], original.loc[k, f"{metric}_std"]
                ok = len(values) == 10 and abs(mean - expected_mean) <= 1e-12 and abs(sd - expected_sd) <= 1e-12
                if metric == "target_f1":
                    ok = ok and abs(mean - rounded[k][0]) <= 5e-7 and abs(sd - rounded[k][1]) <= 5e-7
                    self.log(f"FEWSHOT K={k} target_f1={mean:.6f}±{sd:.6f} n={len(values)} verified={ok}")
                summaries.append({"k": k, "metric": metric, "n": len(values), "mean": mean, "sample_sd": sd,
                                  "saved_mean": expected_mean, "saved_sample_sd": expected_sd, "verified": bool(ok),
                                  "variability": "between-target-family task variability; no optimal-K claim"})
        self.save("fewshot_target_family_f1.csv", wide)
        self.save("fewshot_summary_verified.csv", summaries)
        self.save("fewshot_support_nesting.csv", nesting)
        table = "| Target family | K=1 F1 | K=5 F1 | K=10 F1 |\n|---|---:|---:|---:|\n"
        table += "\n".join(f"| {r['target_family']} | {r['target_f1_K1']:.6f} | {r['target_f1_K5']:.6f} | {r['target_f1_K10']:.6f} |" for r in wide)
        (self.out / "fewshot_manuscript_table.md").write_text(table + "\n\nAcross-family variability does not establish statistical superiority or an optimal support size.\n", encoding="utf-8")
        self.status["FEWSHOT_PER_FAMILY"] = "PASS" if len(wide) == 10 and all(r["verified"] for r in summaries) else "FAIL"
        self.status["FEWSHOT_SUPPORT_NESTING"] = "PASS" if len(nesting) == 30 and all(r["status"] == "PASS" for r in nesting) else "FAIL"

    def feature_audit(self):
        paths = {name: self.root / path for name, path in {
            "extraction": "scripts/phase2/extract_ustc_flows.py", "controller": "controller/phase4_monitor.py",
            "l2": "controller/l2_learning.py", "closed": "scripts/phase3/run_ustc_closed_set.py",
            "open": "scripts/phase3/run_ustc_open_set.py", "service": "scripts/phase4/run_inference_service.py"}.items()}
        original_sources = {k: self.use(p).read_text() for k, p in paths.items()}
        source = {k: ast.unparse(ast.parse(text)) for k, text in original_sources.items()}
        def has(key, *snippets):
            return all(s in source[key] for s in snippets)
        specifications = [
            ("packet_length", "tshark frame.len: observed Ethernet frame bytes", "len(msg.data): full PacketIn frame bytes", "unchanged signed byte sizes",
             has("extraction", "'frame.len'") and has("controller", "len(msg.data)", "direction * frame_length"),
             "Agreement requires full untruncated PacketIn frames and the same link framing; no paired measurement exists."),
            ("direction", "+ first-observed endpoint pair, - reverse", "+ first-observed endpoint pair, - reverse", "preserves sign",
             has("extraction", "direction = 1", "direction = -1", "direction * frame_length") and has("controller", "direction = 1", "direction = -1", "'first_source'", "'first_destination'"), ""),
            ("first_packet_iat", "0.0 seconds", "0.0 seconds", "preserved", has("extraction", "interarrival = 0.0") and has("controller", "iat = 0.0"), ""),
            ("iat_units", "capture timestamp differences in seconds", "monotonic_ns differences divided by 1e9", "seconds; no unit conversion",
             has("extraction", "'frame.time_epoch'", "timestamp - state.last_time") and has("controller", "time.monotonic_ns()", "/ 1000000000.0"), "Equal units do not establish equal timing distributions."),
            ("iat_transform", "log1p(seconds), then train-derived z-score", "raw nonnegative seconds as interarrival_seconds", "same log1p and saved z-score",
             all(has(k, "np.log1p", "'iat_log1p_mean'", "'iat_log1p_std'") for k in ("closed", "open", "service")), ""),
            ("size_transform", "saved training-valid-token RMS; no centering", "raw signed frame size", "same checkpoint RMS",
             all(has(k, "np.float32(normalization['signed_size_rms'])") for k in ("closed", "open")) and has("service", "np.float32(self.normalization['signed_size_rms'])"), ""),
            ("feature_order", "[signed_size, normalized_log1p_iat]", "separate named size/IAT fields", "same two-channel order",
             has("controller", "'signed_packet_sizes': signed_sizes", "'interarrival_seconds': iats"), "Encoder equality tested numerically on prepared records."),
            ("padding_mask", "float32 [50,2], zero padding, convolution mask, packed LSTM", "first 5 valid packets sent", "float32 [1,50,2], length=5, original Encoder",
             has("service", "(1, 50, 2)", "from run_ustc_open_set import Encoder") and has("open", "pack_padded_sequence", "features = features * mask"), "Short flows excluded from matched comparisons."),
            ("precision", "sizes float32, log1p IAT arithmetic float64, tensors float32, lengths int64", "Python integer sizes and float seconds as JSON", "same arithmetic precision and tensor dtypes",
             all(has(k, "dtype=np.float32", "dtype=np.float64", "np.log1p") for k in ("closed", "open", "service")), ""),
        ]
        ck = self.checkpoint(self.path(self.deployment["inference"]["model"]))
        runtime = self.service.Runtime.__new__(self.service.Runtime)
        runtime.normalization, runtime.device = ck["normalization"], self.device
        max_error, checks = 0.0, 0
        for _, group in self.frame[self.frame.sequence_packets >= 5].groupby("class_name", sort=True):
            chosen = group.iloc[np.linspace(0, len(group) - 1, min(16, len(group)), dtype=int)].copy()
            for horizon in (50, 5):
                x, lengths = self.opened.encode_inputs(chosen, ck["normalization"])
                closed_x, closed_l, _ = self.closed.encode_frame(chosen, ck["normalization"], mapping(self.families))
                if horizon == 5:
                    x[:, 5:, :], closed_x[:, 5:, :] = 0, 0
                    lengths[:], closed_l[:] = 5, 5
                if not np.array_equal(x, closed_x) or not np.array_equal(lengths, closed_l):
                    raise ValueError("Original closed/open feature encoders disagree")
                for i, row in enumerate(chosen.itertuples(index=False)):
                    n = int(lengths[i])
                    sx, sl = runtime._encode(row.signed_packet_sizes[:n], row.interarrival_seconds[:n])
                    error = float(np.max(np.abs(sx.cpu().numpy()[0] - x[i])))
                    max_error = max(max_error, error)
                    if error != 0 or int(sl[0]) != n:
                        raise ValueError(f"Service/original encoder mismatch flow={row.flow_id} horizon={horizon} error={error}")
                    checks += 1
        rows = [{"feature": feature, "offline": offline, "controller": controller, "service": service,
                 "status": "PASS" if ok else "FAIL", "limitation": limitation} for feature, offline, controller, service, ok, limitation in specifications]
        rows.extend([
            {"feature": "clock_source", "offline": "PCAP acquisition timestamps", "controller": "controller handler monotonic arrival timestamps",
             "service": "uses supplied IATs", "status": "DIFFERENT", "limitation": "Control-plane transport, queuing and scheduling can alter timing; not measured."},
            {"feature": "encoder_numeric_equivalence", "offline": "closed/open original encoders", "controller": "raw feature contract only",
             "service": "actual Runtime._encode", "status": "PASS", "limitation": f"max_abs_difference={max_error}; checks={checks}; no controller timestamp measurement"},
            {"feature": "controller_source", "offline": "Phase2 extraction", "controller": "phase4_monitor.py collects features; l2_learning.py only forwards",
             "service": "scripts/phase4/run_inference_service.py", "status": "PASS", "limitation": ""},
        ])
        source_terms = {
            "packet_length": ("frame.len", "frame_length", "msg.data"), "direction": ("direction", "first_source", "first_destination"),
            "first_packet_iat": ("interarrival = 0.0", "iat = 0.0"), "iat_units": ("frame.time_epoch", "monotonic_ns", "1_000_000_000.0"),
            "iat_transform": ("log1p", "iat_log1p_mean", "iat_log1p_std"), "size_transform": ("signed_size_rms",),
            "feature_order": ("signed_packet_sizes", "interarrival_seconds"), "padding_mask": ("maximum_length", "pack_padded_sequence", "features = features * mask"),
            "precision": ("dtype=np.float32", "dtype=np.float64", "dtype=torch.int64"), "clock_source": ("frame.time_epoch", "monotonic_ns"),
        }
        for row in rows:
            evidence = []
            for key, text in original_sources.items():
                selected = [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1)
                            if any(term in line for term in source_terms.get(row["feature"], ()))]
                evidence.extend(f"{paths[key].relative_to(self.root)}:{i}: {line}" for i, line in selected[:8])
            row["source_evidence"] = " | ".join(evidence)
        self.save("feature_compatibility_audit.csv", rows)
        self.status["FEATURE_CODE_CONSISTENCY"] = "PASS" if all(ok for _, _, _, _, ok, _ in specifications) else "FAIL"
        self.details["feature_consistency_scope"] = "Definitions, units, transforms and precision agree conditionally on full frame delivery. Clock sources differ. This status does not verify timing-distribution agreement."
        self.log(f"FEATURE_ENCODER_EQUIVALENCE checks={checks} max_abs_difference={max_error} clock_sources=DIFFERENT measured_timing=NOT_VERIFIED")

    def timing_audit(self):
        folder = self.root / "artifacts/phase4"
        pcaps = sorted(str(p.relative_to(self.root)) for p in folder.rglob("*") if p.suffix.lower() in (".pcap", ".pcapng", ".cap"))
        telemetry = []
        for path in sorted(folder.rglob("*metrics.jsonl")):
            fields, count = set(), 0
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        fields.update(json.loads(line))
                        count += 1
            telemetry.append({"path": str(path.relative_to(self.root)), "records": count, "fields": sorted(fields)})
        self.details["controller_pcap_timing"] = {
            "status": "NOT_VERIFIED", "phase4_capture_files": pcaps, "telemetry": telemetry,
            "reason": "Phase4 telemetry stores flow-level overhead and predictions without packet arrays, per-packet timestamps or matchable packet identities; no paired Phase4 capture exists.",
            "required_observations": [
                "Capture the same packets at documented switch ingress; record timestamp resolution and full Ethernet lengths.",
                "Controller log per PacketIn: monotonic_ns, switch DPID/port, canonical 5-tuple, flow-instance/reset identity, sequence rank, frame bytes/hash, signed length and direction.",
                "Match unambiguously by flow instance and frame identity; check packet order and completeness including retransmissions and duplicates.",
                "Compare first-five packet size/order/direction, IAT median/p95/p99, absolute IAT differences and normalized feature differences using saved training statistics. Absolute timestamps require synchronized clocks; IAT differences do not.",
            ],
            "capture_mode": "Not implemented: controller per-packet observations are absent. The audit starts no Mininet, switch changes, controller or capture.",
        }
        self.status["CONTROLLER_PCAP_IAT_AGREEMENT"] = "NOT_VERIFIED"
        self.log(f"CONTROLLER_PCAP_IAT_AGREEMENT=NOT_VERIFIED phase4_capture_files={len(pcaps)} telemetry_files={len(telemetry)}; flow overhead is not packet timestamp agreement")

    def finish(self):
        after = self.frozen_snapshot()
        changed = sorted(name for name in set(self.before) | set(after) if self.before.get(name) != after.get(name))
        self.status["ORIGINAL_ARTIFACTS_UNCHANGED"] = "PASS" if not changed else "FAIL"
        self.details["original_artifacts"] = {"checked_files": len(self.before), "changed_paths": changed}
        (self.out / "original_artifact_hashes.json").write_text(json.dumps(self.before, indent=2) + "\n", encoding="utf-8")
        try:
            self.verify_manifests()
        except Exception as exc:
            self.error("frozen manifest verification", exc)
        failures = any(value == "FAIL" for value in self.status.values()) or bool(self.errors)
        final = "FAIL" if failures else ("PARTIAL" if "NOT_VERIFIED" in self.status.values() else "PASS")
        for name in STATUS_NAMES:
            self.log(f"{name}={self.status[name]}")
        self.log(f"FINAL_COMMENT3_4_AUDIT={final}")
        self.log(f"OUTPUT_DIRECTORY={self.out}")
        filenames = (
            "closed_five_packet_per_seed.csv", "closed_five_packet_summary.csv", "open_five_packet_per_rotation.csv",
            "open_five_packet_summary.csv", "deployment_run0_five_packet.csv", "five_packet_threshold_comparison.csv",
            "fewshot_target_family_f1.csv", "fewshot_summary_verified.csv", "feature_compatibility_audit.csv",
        )
        for name in filenames:
            if name not in self.rows:
                self.save(name, [{"status": "NOT_VERIFIED", "reason": "Evaluation failed; see audit_log.txt and final_audit_summary.json; no metric substituted."}])
        summary = {"final_status": final, "requirements": self.status, "details": self.details, "errors": self.errors,
                   "used_frozen_artifacts": self.used, "device": str(self.device),
                   "tolerances": {"metric_absolute": METRIC_ATOL, "probability_absolute": PROBABILITY_ATOL, "prediction_mismatch_rate": PREDICTION_MISMATCH_RATE,
                                  "ap_rank_score_epsilon": float(np.finfo(np.float32).eps)},
                   "metrics": self.rows, "limitations": ["Controller/PCAP timing agreement is unmeasured.",
                       "Offline USTC first-five performance does not measure accuracy on synthetic Phase4 traffic.",
                       "Original open-set per-flow score artifacts were not saved; reproduction uses per-rotation reports."]}
        (self.out / "final_audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 2 if failures else 0

    def run(self):
        try:
            self.configure()
            self.load_data()
        except Exception as exc:
            self.error("dataset discovery/partition validation", exc)
        if hasattr(self, "frame"):
            for name, function in (("fewshot", self.fewshot_audit), ("closed", self.closed_audit), ("open", self.open_audit),
                                   ("deployment", self.deployment_audit), ("feature", self.feature_audit)):
                try:
                    function()
                except Exception as exc:
                    self.error(name, exc)
                    affected = {"fewshot": ["FEWSHOT_PER_FAMILY", "FEWSHOT_SUPPORT_NESTING"],
                                "closed": ["CLOSED_ORIGINAL_REPRODUCTION", "CLOSED_N5_EVALUATION"],
                                "open": ["OPEN_ORIGINAL_REPRODUCTION", "OPEN_N5_EVALUATION", "FIVE_PACKET_THRESHOLD_COMPARISON"],
                                "deployment": ["DEPLOYMENT_RUN0_N5"], "feature": ["FEATURE_CODE_CONSISTENCY"]}
                    for status in affected[name]:
                        self.status[status] = "FAIL"
        try:
            self.timing_audit()
        except Exception as exc:
            self.error("Phase4 paired timing discovery", exc)
        return self.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", help="legacy positional repository root")
    parser.add_argument("--root", dest="root_option", help="repository root (default: .)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4, help="CPU inference threads")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("--batch-size and --threads must be positive")
    try:
        return Audit(args).run()
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
