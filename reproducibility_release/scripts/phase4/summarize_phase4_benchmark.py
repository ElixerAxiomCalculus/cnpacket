import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DIRECTORY = (
    ROOT
    / "artifacts"
    / "phase4"
    / "benchmark"
)

OUTPUT_PATH = (
    DIRECTORY
    / "phase4_benchmark_summary.json"
)

CSV_PATH = (
    DIRECTORY
    / "phase4_pipeline_summary.csv"
)


def describe(
    values
):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(
        values
    ) == 0:
        return None

    return {
        "count": int(
            len(
                values
            )
        ),
        "mean": float(
            values.mean()
        ),
        "std": float(
            values.std(
                ddof=1
            )
        ),
        "p50": float(
            np.percentile(
                values,
                50
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95
            )
        ),
        "p99": float(
            np.percentile(
                values,
                99
            )
        ),
        "min": float(
            values.min()
        ),
        "max": float(
            values.max()
        )
    }


def read_jsonl(
    path
):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(
                    line
                )
            )

    return rows


def main():
    modes = [
        "forwarding",
        "features",
        "inference"
    ]

    benchmark = {}

    for mode in modes:
        path = (
            DIRECTORY
            / f"{mode}_benchmark.json"
        )

        benchmark[
            mode
        ] = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    network_rows = []

    for mode in modes:
        rows = benchmark[
            mode
        ][
            "network_trials"
        ]

        network_rows.append(
            {
                "mode": mode,
                "trials": len(
                    rows
                ),
                "packet_loss_mean_percent": float(
                    np.mean(
                        [
                            row[
                                "packet_loss_percent"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "ping_rtt_ms_mean": float(
                    np.mean(
                        [
                            row[
                                "ping_average_rtt_ms"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "ping_rtt_ms_std": float(
                    np.std(
                        [
                            row[
                                "ping_average_rtt_ms"
                            ]
                            for row in rows
                        ],
                        ddof=1
                    )
                ),
                "throughput_mbit_s_mean": float(
                    np.mean(
                        [
                            row[
                                "iperf_throughput_mbit_s"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "throughput_mbit_s_std": float(
                    np.std(
                        [
                            row[
                                "iperf_throughput_mbit_s"
                            ]
                            for row in rows
                        ],
                        ddof=1
                    )
                ),
                "microflow_rtt_ms_mean": float(
                    benchmark[
                        mode
                    ][
                        "microflows"
                    ][
                        "latency_ms_mean"
                    ]
                ),
                "microflow_rtt_ms_std": float(
                    benchmark[
                        mode
                    ][
                        "microflows"
                    ][
                        "latency_ms_std"
                    ]
                ),
                "microflow_rtt_ms_p50": float(
                    benchmark[
                        mode
                    ][
                        "microflows"
                    ][
                        "latency_ms_p50"
                    ]
                ),
                "microflow_rtt_ms_p95": float(
                    benchmark[
                        mode
                    ][
                        "microflows"
                    ][
                        "latency_ms_p95"
                    ]
                ),
                "microflow_rtt_ms_p99": float(
                    benchmark[
                        mode
                    ][
                        "microflows"
                    ][
                        "latency_ms_p99"
                    ]
                )
            }
        )

    network = pd.DataFrame(
        network_rows
    )

    feature_rows = read_jsonl(
        DIRECTORY
        / "features_controller_metrics.jsonl"
    )

    inference_rows = read_jsonl(
        DIRECTORY
        / "inference_controller_metrics.jsonl"
    )

    if len(
        feature_rows
    ) < 500:
        raise RuntimeError(
            f"Expected >=500 feature records, "
            f"found {len(feature_rows)}"
        )

    if len(
        inference_rows
    ) < 500:
        raise RuntimeError(
            f"Expected >=500 inference records, "
            f"found {len(inference_rows)}"
        )

    inference_errors = [
        row
        for row in inference_rows
        if "inference_error" in row
    ]

    if inference_errors:
        raise RuntimeError(
            f"Inference errors found: "
            f"{len(inference_errors)}"
        )

    pipeline = {
        "features": {
            "records": len(
                feature_rows
            ),
            "feature_total_us": describe(
                [
                    row[
                        "feature_total_us"
                    ]
                    for row in feature_rows
                ]
            ),
            "feature_last_packet_us": describe(
                [
                    row[
                        "feature_last_packet_us"
                    ]
                    for row in feature_rows
                ]
            ),
            "controller_decision_us": describe(
                [
                    row[
                        "controller_decision_us"
                    ]
                    for row in feature_rows
                ]
            )
        },
        "inference": {
            "records": len(
                inference_rows
            ),
            "feature_total_us": describe(
                [
                    row[
                        "feature_total_us"
                    ]
                    for row in inference_rows
                ]
            ),
            "feature_last_packet_us": describe(
                [
                    row[
                        "feature_last_packet_us"
                    ]
                    for row in inference_rows
                ]
            ),
            "inference_roundtrip_us": describe(
                [
                    row[
                        "inference_roundtrip_us"
                    ]
                    for row in inference_rows
                ]
            ),
            "service_us": describe(
                [
                    row[
                        "service_us"
                    ]
                    for row in inference_rows
                ]
            ),
            "model_us": describe(
                [
                    row[
                        "model_us"
                    ]
                    for row in inference_rows
                ]
            ),
            "controller_decision_us": describe(
                [
                    row[
                        "controller_decision_us"
                    ]
                    for row in inference_rows
                ]
            )
        }
    }

    forwarding_row = network[
        network[
            "mode"
        ]
        == "forwarding"
    ].iloc[
        0
    ]

    overhead = {}

    for mode in [
        "features",
        "inference"
    ]:
        row = network[
            network[
                "mode"
            ]
            == mode
        ].iloc[
            0
        ]

        overhead[
            mode
        ] = {
            "ping_rtt_delta_ms": float(
                row[
                    "ping_rtt_ms_mean"
                ]
                - forwarding_row[
                    "ping_rtt_ms_mean"
                ]
            ),
            "throughput_delta_mbit_s": float(
                row[
                    "throughput_mbit_s_mean"
                ]
                - forwarding_row[
                    "throughput_mbit_s_mean"
                ]
            ),
            "throughput_delta_percent": float(
                (
                    row[
                        "throughput_mbit_s_mean"
                    ]
                    - forwarding_row[
                        "throughput_mbit_s_mean"
                    ]
                )
                / forwarding_row[
                    "throughput_mbit_s_mean"
                ]
                * 100.0
            ),
            "microflow_p50_delta_ms": float(
                row[
                    "microflow_rtt_ms_p50"
                ]
                - forwarding_row[
                    "microflow_rtt_ms_p50"
                ]
            ),
            "microflow_p95_delta_ms": float(
                row[
                    "microflow_rtt_ms_p95"
                ]
                - forwarding_row[
                    "microflow_rtt_ms_p95"
                ]
            ),
            "microflow_p99_delta_ms": float(
                row[
                    "microflow_rtt_ms_p99"
                ]
                - forwarding_row[
                    "microflow_rtt_ms_p99"
                ]
            )
        }

    output = {
        "network": network.to_dict(
            orient="records"
        ),
        "pipeline": pipeline,
        "overhead_vs_forwarding": overhead,
        "interpretation": {
            "pipeline_latency_reference": (
                "processing after the fifth observed packet"
            ),
            "synthetic_accuracy_claim": False
        }
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2
        )
        + "\n",
        encoding="utf-8"
    )

    csv_rows = []

    for mode, values in pipeline.items():
        for metric, summary in values.items():
            if metric == "records":
                continue

            csv_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    **summary
                }
            )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        CSV_PATH,
        index=False
    )

    print(
        "===== NETWORK SUMMARY ====="
    )

    print(
        network.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PIPELINE SUMMARY ====="
    )

    for mode, values in pipeline.items():
        print()
        print(
            f"[{mode}] "
            f"records={values['records']}"
        )

        for metric, summary in values.items():
            if metric == "records":
                continue

            print(
                f"{metric}: "
                f"mean={summary['mean']:.3f} "
                f"std={summary['std']:.3f} "
                f"p50={summary['p50']:.3f} "
                f"p95={summary['p95']:.3f} "
                f"p99={summary['p99']:.3f} us"
            )

    print()
    print(
        "===== OVERHEAD VS FORWARDING ====="
    )

    for mode, values in overhead.items():
        print(
            f"{mode}: "
            f"RTT_DELTA_MS="
            f"{values['ping_rtt_delta_ms']:.6f} "
            f"THROUGHPUT_DELTA_MBIT_S="
            f"{values['throughput_delta_mbit_s']:.6f} "
            f"THROUGHPUT_DELTA_PERCENT="
            f"{values['throughput_delta_percent']:.6f} "
            f"MICROFLOW_P50_DELTA_MS="
            f"{values['microflow_p50_delta_ms']:.6f} "
            f"MICROFLOW_P95_DELTA_MS="
            f"{values['microflow_p95_delta_ms']:.6f} "
            f"MICROFLOW_P99_DELTA_MS="
            f"{values['microflow_p99_delta_ms']:.6f}"
        )

    print()
    print(
        f"SUMMARY_JSON={OUTPUT_PATH}"
    )

    print(
        f"PIPELINE_CSV={CSV_PATH}"
    )

    print(
        "PHASE4_BENCHMARK_SUMMARY=PASS"
    )


if __name__ == "__main__":
    main()
