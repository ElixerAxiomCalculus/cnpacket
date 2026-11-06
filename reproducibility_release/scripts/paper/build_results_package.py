from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "artifacts"
    / "paper"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_ROWS = [
    {
        "stage": "3A",
        "dataset": "CICIDS2017",
        "experiment": "Random stratified split",
        "method": "Random Forest",
        "metric": "Macro-F1",
        "mean": 0.998078,
        "std": 0.000047,
        "unit": "fraction",
        "source": "Phase 3A.1 frozen results"
    },
    {
        "stage": "3A",
        "dataset": "CICIDS2017",
        "experiment": "Random stratified split",
        "method": "XGBoost",
        "metric": "Macro-F1",
        "mean": 0.998888,
        "std": 0.000016,
        "unit": "fraction",
        "source": "Phase 3A.1 frozen results"
    },
    {
        "stage": "3A",
        "dataset": "CICIDS2017",
        "experiment": "Capture-day holdout",
        "method": "Random Forest",
        "metric": "Equal-day Macro-F1",
        "mean": 0.529178,
        "std": np.nan,
        "unit": "fraction",
        "source": "Phase 3A.2 frozen results"
    },
    {
        "stage": "3A",
        "dataset": "CICIDS2017",
        "experiment": "Capture-day holdout",
        "method": "XGBoost",
        "metric": "Equal-day Macro-F1",
        "mean": 0.500492,
        "std": np.nan,
        "unit": "fraction",
        "source": "Phase 3A.2 frozen results"
    },
    {
        "stage": "3B",
        "dataset": "ISCXVPN2016",
        "experiment": "Capture-fold sequence classification",
        "method": "CNN-BiLSTM",
        "metric": "Macro-F1",
        "mean": 0.375958,
        "std": 0.078866,
        "unit": "fraction",
        "source": "Phase 3B.1 frozen results"
    },
    {
        "stage": "3B",
        "dataset": "ISCXVPN2016",
        "experiment": "Common-cohort early sequence N=20",
        "method": "CNN-BiLSTM",
        "metric": "Macro-F1",
        "mean": 0.467906,
        "std": 0.131010,
        "unit": "fraction",
        "source": "Phase 3B.2 frozen results"
    },
    {
        "stage": "3C",
        "dataset": "USTC-TFC2016",
        "experiment": "10-family closed set",
        "method": "CNN-BiLSTM",
        "metric": "Macro-F1",
        "mean": 0.738051,
        "std": 0.012824,
        "unit": "fraction",
        "source": "Phase 3C frozen results"
    },
    {
        "stage": "3C",
        "dataset": "USTC-TFC2016",
        "experiment": "10-family closed set",
        "method": "CNN-BiLSTM",
        "metric": "Balanced accuracy",
        "mean": 0.806591,
        "std": 0.012681,
        "unit": "fraction",
        "source": "Phase 3C frozen results"
    },
    {
        "stage": "3C",
        "dataset": "USTC-TFC2016",
        "experiment": "10-family closed set",
        "method": "CNN-BiLSTM",
        "metric": "Macro ROC-AUC",
        "mean": 0.962743,
        "std": 0.005594,
        "unit": "fraction",
        "source": "Phase 3C frozen results"
    },
    {
        "stage": "3D",
        "dataset": "USTC-TFC2016",
        "experiment": "Zero-shot open set",
        "method": "CE + MSP",
        "metric": "Unknown AUROC",
        "mean": 0.625537,
        "std": 0.105720,
        "unit": "fraction",
        "source": "artifacts/phase3/ustc/open_set/zero_shot_summary.csv"
    },
    {
        "stage": "3D",
        "dataset": "USTC-TFC2016",
        "experiment": "Zero-shot open set",
        "method": "SupCon + prototype",
        "metric": "Unknown AUROC",
        "mean": 0.652948,
        "std": 0.191855,
        "unit": "fraction",
        "source": "artifacts/phase3/ustc/open_set/zero_shot_summary.csv"
    },
    {
        "stage": "3D",
        "dataset": "USTC-TFC2016",
        "experiment": "Zero-shot open set",
        "method": "CE + MSP",
        "metric": "Open-set Macro-F1",
        "mean": 0.557808,
        "std": 0.129792,
        "unit": "fraction",
        "source": "artifacts/phase3/ustc/open_set/zero_shot_summary.csv"
    },
    {
        "stage": "3D",
        "dataset": "USTC-TFC2016",
        "experiment": "Zero-shot open set",
        "method": "SupCon + prototype",
        "metric": "Open-set Macro-F1",
        "mean": 0.585035,
        "std": 0.160692,
        "unit": "fraction",
        "source": "artifacts/phase3/ustc/open_set/zero_shot_summary.csv"
    },
    {
        "stage": "3D",
        "dataset": "USTC-TFC2016",
        "experiment": "Few-shot adaptation K=5",
        "method": "SupCon + prototype",
        "metric": "Nine-family Macro-F1",
        "mean": 0.674043,
        "std": 0.063215,
        "unit": "fraction",
        "source": "artifacts/phase3/ustc/open_set/few_shot_summary.csv"
    },
    {
        "stage": "3D",
        "dataset": "USTC-TFC2016",
        "experiment": "Few-shot adaptation K=5",
        "method": "SupCon + prototype",
        "metric": "Target-family F1",
        "mean": 0.439168,
        "std": 0.301118,
        "unit": "fraction",
        "source": "artifacts/phase3/ustc/open_set/few_shot_summary.csv"
    },
    {
        "stage": "4",
        "dataset": "Mininet",
        "experiment": "SDN feature path",
        "method": "Feature collection",
        "metric": "Controller decision latency",
        "mean": 107.975,
        "std": 41.972,
        "unit": "microseconds",
        "source": "artifacts/phase4/benchmark/phase4_benchmark_summary.json"
    },
    {
        "stage": "4",
        "dataset": "Mininet",
        "experiment": "SDN inference path",
        "method": "SupCon inference service",
        "metric": "Controller decision latency",
        "mean": 3810.099,
        "std": 839.272,
        "unit": "microseconds",
        "source": "artifacts/phase4/benchmark/phase4_benchmark_summary.json"
    },
    {
        "stage": "4",
        "dataset": "Mininet",
        "experiment": "SDN inference path",
        "method": "CNN-BiLSTM encoder",
        "metric": "Model inference latency",
        "mean": 1485.059,
        "std": 368.098,
        "unit": "microseconds",
        "source": "artifacts/phase4/benchmark/phase4_benchmark_summary.json"
    },
    {
        "stage": "4",
        "dataset": "Mininet",
        "experiment": "SDN inference path",
        "method": "SupCon inference service",
        "metric": "Incremental GPU memory",
        "mean": 233.0,
        "std": 0.0,
        "unit": "MiB",
        "source": "artifacts/phase4/benchmark/gpu_memory_summary.json"
    }
]


CLAIMS = [
    {
        "claim": "Random-split flow classification substantially overestimates robustness under capture-context shift.",
        "evidence": "CICIDS2017 random-split Macro-F1 exceeded 0.998, while equal-day holdout Macro-F1 was about 0.50-0.53 and attack recall collapsed on several held-out days.",
        "allowed_wording": "Random splits substantially overestimate cross-context performance in this dataset.",
        "limitation": "The day protocol represents capture-context/scenario shift rather than pure chronological generalization."
    },
    {
        "claim": "CNN-BiLSTM was the strongest observed ISCXVPN2016 architecture.",
        "evidence": "CNN-BiLSTM Macro-F1 was 0.376 ± 0.079 across capture folds, slightly above CNN, LSTM, and BiLSTM.",
        "allowed_wording": "CNN-BiLSTM achieved the highest observed mean Macro-F1 among the evaluated architectures.",
        "limitation": "Differences were modest and fold variability was large."
    },
    {
        "claim": "Intermediate packet prefixes can outperform longer prefixes.",
        "evidence": "On the common >=50-packet cohort, N=20 achieved 0.468 ± 0.131 Macro-F1, above N=5, N=10, and N=50.",
        "allowed_wording": "N=20 gave the highest observed mean on the common cohort.",
        "limitation": "No universal optimum or statistical significance is claimed."
    },
    {
        "claim": "Closed-set malware-family discrimination is much easier than unseen-family detection.",
        "evidence": "USTC closed-set Macro ROC-AUC was 0.963 ± 0.006 and Macro-F1 0.738 ± 0.013, while zero-shot unknown AUROC was only 0.626 for CE and 0.653 for SupCon.",
        "allowed_wording": "Strong closed-set discrimination did not translate into equally strong open-set detection.",
        "limitation": "Most USTC malware families originate from a single capture, so closed-set results are not independent-capture generalization."
    },
    {
        "claim": "SupCon improves representation quality but does not uniformly solve open-set rejection.",
        "evidence": "SupCon improved mean unknown AUROC by 0.027 and open-set Macro-F1 by 0.027, but won only 6 of 10 target families on both metrics.",
        "allowed_wording": "SupCon provided modest average gains with strong target-family dependence.",
        "limitation": "Ten rotations represent family/task variability rather than stochastic-seed uncertainty."
    },
    {
        "claim": "SupCon trades lower known false rejection against lower unknown recall.",
        "evidence": "Known FRR decreased from 0.544 to 0.330, while unknown recall decreased from 0.853 to 0.638.",
        "allowed_wording": "The proposed representation reduced known rejection but did not dominate the baseline on unknown recall.",
        "limitation": "Performance depends strongly on threshold transfer between calibration and target unknown families."
    },
    {
        "claim": "Few-shot prototype adaptation is useful on average but family dependent.",
        "evidence": "K=5 gave the highest observed target-family F1, 0.439 ± 0.301, but Cridex collapsed while Virut and Nsis-ay improved strongly.",
        "allowed_wording": "Five-shot adaptation gave the highest observed average target F1, with substantial family-specific variation.",
        "limitation": "More support samples can dilute a single prototype when support embeddings are heterogeneous."
    },
    {
        "claim": "The SDN deployment introduces millisecond-scale inference latency without material throughput loss in the tested environment.",
        "evidence": "Inference controller decision latency was 3.81 ms mean, 5.64 ms p95, and 6.67 ms p99; throughput changed from 94.76 to 94.64 Mbit/s with zero packet loss.",
        "allowed_wording": "The tested SDN deployment incurred millisecond-scale decision latency while showing negligible throughput degradation under the benchmark workload.",
        "limitation": "The workload is a controlled Mininet benchmark, not Internet-scale deployment."
    },
    {
        "claim": "Mininet traffic validates system overhead but not malware-classification accuracy.",
        "evidence": "Synthetic ping, iperf, and TCP microflows have no USTC malware ground-truth labels.",
        "allowed_wording": "Phase 4 measures deployment overhead only.",
        "limitation": "Classification accuracy claims must remain tied to the frozen labeled datasets."
    }
]


CLOSED_F1 = {
    "Zeus": 0.975261,
    "Shifu": 0.967685,
    "Virut": 0.962137,
    "Neris": 0.871932,
    "Nsis-ay": 0.834017,
    "Tinba": 0.743683,
    "Cridex": 0.655839,
    "Miuref": 0.579192,
    "Geodo": 0.434050,
    "Htbot": 0.356716
}


TARGETS = [
    "Cridex",
    "Geodo",
    "Htbot",
    "Miuref",
    "Neris",
    "Nsis-ay",
    "Shifu",
    "Tinba",
    "Virut",
    "Zeus"
]

CE_AUROC = [
    0.503558,
    0.801396,
    0.671996,
    0.602660,
    0.693652,
    0.582396,
    0.617980,
    0.708753,
    0.429809,
    0.643175
]

SUPCON_AUROC = [
    0.523146,
    0.507689,
    0.911302,
    0.529263,
    0.500447,
    0.891410,
    0.896644,
    0.781962,
    0.541130,
    0.446486
]


FEWSHOT_K = [
    1,
    5,
    10
]

FEWSHOT_TARGET_F1_MEAN = [
    0.296120,
    0.439168,
    0.422812
]

FEWSHOT_TARGET_F1_STD = [
    0.263053,
    0.301118,
    0.312041
]


LATENCY_CATEGORIES = [
    "Features\ncontroller",
    "GPU model",
    "Inference\ncontroller"
]

LATENCY_P50 = [
    97.841,
    1389.683,
    3629.372
]

LATENCY_P95 = [
    182.875,
    1945.151,
    5636.088
]

LATENCY_P99 = [
    241.413,
    2422.988,
    6671.421
]


def save_master_tables():
    pd.DataFrame(
        MASTER_ROWS
    ).to_csv(
        OUTPUT / "master_results.csv",
        index=False
    )

    pd.DataFrame(
        CLAIMS
    ).to_csv(
        OUTPUT / "claims_matrix.csv",
        index=False
    )


def closed_set_figure():
    frame = pd.DataFrame(
        {
            "family": list(
                CLOSED_F1.keys()
            ),
            "f1": list(
                CLOSED_F1.values()
            )
        }
    ).sort_values(
        "f1",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            5.2
        )
    )

    ax.barh(
        frame[
            "family"
        ],
        frame[
            "f1"
        ]
    )

    ax.set_xlabel(
        "F1 score"
    )

    ax.set_ylabel(
        "Malware family"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_title(
        "USTC-TFC2016 closed-set per-family F1"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "figure1_closed_set_family_f1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def open_set_figure():
    positions = np.arange(
        len(
            TARGETS
        )
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            4.8
        )
    )

    ax.bar(
        positions
        - width / 2,
        CE_AUROC,
        width,
        label="CE + MSP"
    )

    ax.bar(
        positions
        + width / 2,
        SUPCON_AUROC,
        width,
        label="SupCon + prototype"
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        TARGETS,
        rotation=35,
        ha="right"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Unknown-detection AUROC"
    )

    ax.set_xlabel(
        "Held-out malware family"
    )

    ax.set_title(
        "Zero-shot unknown detection by target family"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "figure2_open_set_auroc.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def fewshot_figure():
    fig, ax = plt.subplots(
        figsize=(
            6.8,
            4.6
        )
    )

    ax.errorbar(
        FEWSHOT_K,
        FEWSHOT_TARGET_F1_MEAN,
        yerr=FEWSHOT_TARGET_F1_STD,
        marker="o",
        capsize=5
    )

    ax.set_xticks(
        FEWSHOT_K
    )

    ax.set_ylim(
        0,
        0.8
    )

    ax.set_xlabel(
        "Support examples per target family"
    )

    ax.set_ylabel(
        "Target-family F1"
    )

    ax.set_title(
        "Few-shot adaptation across unseen malware families"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "figure3_fewshot_target_f1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def latency_figure():
    positions = np.arange(
        len(
            LATENCY_CATEGORIES
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(
            7.8,
            4.8
        )
    )

    ax.bar(
        positions
        - width,
        LATENCY_P50,
        width,
        label="p50"
    )

    ax.bar(
        positions,
        LATENCY_P95,
        width,
        label="p95"
    )

    ax.bar(
        positions
        + width,
        LATENCY_P99,
        width,
        label="p99"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        LATENCY_CATEGORIES
    )

    ax.set_ylabel(
        "Latency (µs)"
    )

    ax.set_title(
        "Phase-4 processing latency distribution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "figure4_sdn_latency.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def write_digest():
    text = """# Consolidated Results

## Static-flow robustness

CICIDS2017 random-split classification produced near-perfect Macro-F1 for Random Forest and XGBoost, but capture-day holdout reduced equal-day Macro-F1 to approximately 0.53 and 0.50 respectively. The result demonstrates substantial sensitivity to capture-context shift despite strong in-distribution performance.

## Encrypted sequence classification

On ISCXVPN2016, CNN-BiLSTM achieved the highest observed architecture-level mean Macro-F1 of 0.376 ± 0.079. On a common >=50-packet cohort, the 20-packet prefix produced the highest observed mean Macro-F1 of 0.468 ± 0.131, indicating that longer observation windows did not monotonically improve performance.

## Closed-set malware classification

The USTC CNN-BiLSTM achieved Macro-F1 0.738 ± 0.013, balanced accuracy 0.807 ± 0.013, and macro ROC-AUC 0.963 ± 0.006. Performance varied substantially by malware family, with Zeus, Shifu, and Virut considerably easier than Geodo and Htbot.

## Open-set detection

Across ten held-out-family rotations, CE+MSP achieved unknown AUROC 0.626 ± 0.106 and open-set Macro-F1 0.558 ± 0.130. SupCon+prototype increased these means to 0.653 ± 0.192 and 0.585 ± 0.161. SupCon reduced known false rejection from 0.544 to 0.330 but also reduced unknown recall from 0.853 to 0.638. The effect was strongly family dependent rather than uniformly superior.

## Few-shot adaptation

The highest observed mean target-family F1 occurred at K=5 with 0.439 ± 0.301. Nine-family Macro-F1 was 0.658 ± 0.091 at K=1, 0.674 ± 0.063 at K=5, and 0.672 ± 0.066 at K=10. The large target-family variance and the Cridex geometry audit show that additional support can dilute a single prototype when support samples are heterogeneous.

## SDN deployment

Feature collection required approximately 53.6 µs mean cumulative processing over five observed packets. The complete inference controller path required 3.81 ms mean, 5.64 ms p95, and 6.67 ms p99 after the fifth observed packet. The neural forward pass itself required 1.49 ms mean, 1.95 ms p95, and 2.42 ms p99. Throughput changed from 94.76 Mbit/s in forwarding-only mode to 94.64 Mbit/s with inference, with zero packet loss in all trials. The deployed inference service used approximately 233 MiB incremental whole-device GPU memory.

## Central conclusion

High closed-set accuracy is insufficient evidence of robustness to unseen encrypted malware. Contrastive representation learning modestly improves average open-set performance and often reduces rejection of known traffic, but unseen-family behavior and few-shot prototype quality remain strongly family dependent. The resulting SDN implementation demonstrates that this analysis can be integrated with millisecond-scale controller decision latency and negligible throughput degradation under the tested Mininet workload.
"""

    (
        OUTPUT
        / "results_digest.md"
    ).write_text(
        text,
        encoding="utf-8"
    )


def main():
    save_master_tables()
    closed_set_figure()
    open_set_figure()
    fewshot_figure()
    latency_figure()
    write_digest()

    print(
        "MASTER_RESULTS="
        f"{OUTPUT / 'master_results.csv'}"
    )

    print(
        "CLAIMS_MATRIX="
        f"{OUTPUT / 'claims_matrix.csv'}"
    )

    for index in range(
        1,
        5
    ):
        files = sorted(
            OUTPUT.glob(
                f"figure{index}_*.png"
            )
        )

        for path in files:
            print(
                f"FIGURE={path}"
            )

    print(
        "RESULTS_DIGEST="
        f"{OUTPUT / 'results_digest.md'}"
    )

    print(
        "PAPER_RESULTS_PACKAGE=PASS"
    )


if __name__ == "__main__":
    main()
