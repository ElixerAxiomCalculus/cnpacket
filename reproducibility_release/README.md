# Leakage-Aware Evaluation of Payload-Free Malware Classification Across Unseen Families and SDN Deployment

This code-focused package supports journal-submission reproducibility for the manuscript named above. **Title provenance:** this title was explicitly supplied by the authors in the packaging request; no manuscript file was present in `paper/` or elsewhere in the inspected research documentation to independently verify a newer title. Older environment metadata uses “Deep-Learning Malware/Traffic Classification via SDN”; it is retained unchanged as historical metadata.

The study evaluates how classification changes when capture context, duplicate representations, unseen malware families, and observation length are controlled. Signed packet sizes and inter-arrival times support traffic analysis without inspecting payload bytes. Closed-set classification chooses among families seen in training; open-set evaluation also rejects unseen families. Five-packet evaluation measures the effect of short observation on frozen models, while few-shot adaptation adds a prototype from a small labeled support set. The SDN component measures integration and processing overhead. This is a reproducibility-focused evaluation with substantial generalization limitations.

Packaging date: 6 November 2026 (Asia/Calcutta). No experiment, preprocessing job, inference service, network testbed, or installation was executed during packaging. All copied implementation paths remain relative to this release root.

## Authors and affiliations

| Author | Affiliation | ORCID |
|---|---|---|
| Arush James | Vellore Institute of Technology, Vellore, India | Not provided — to be completed before release |
| Rohan Chaudhary | Vellore Institute of Technology, Vellore, India | Not provided — to be completed before release |
| Sayak Mondal (corresponding author) | Vellore Institute of Technology, Vellore, India | Not provided — to be completed before release |
| B. Kalaavathi | Vellore Institute of Technology, Vellore, India | Not provided — to be completed before release |

Names, affiliations, and correspondence designation are author-provided in the packaging request. No manuscript was available to verify the proposed correspondence email, so it is omitted. No registration numbers or inferred ORCIDs are included.

## Datasets and access

Obtain the original datasets from their providers and follow their access, citation, and redistribution conditions. None is redistributed here. The provider pages below were checked during packaging; local scripts and frozen protocols define the experimental subsets.

| Dataset | Official source | Frozen experimental role and expected local input |
|---|---|---|
| CICIDS2017 | [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | Eight MachineLearningCSV files; 78-feature binary Random Forest/XGBoost baselines, stratified split and capture-day holdout. Archive at `data/raw/cicids2017/MachineLearningCSV.zip`; extract below `data/interim/cicids2017/raw_csv/`. |
| USTC-TFC2016 | [Dataset repository](https://github.com/davidyslu/USTC-TFC2016) | Provider `Benign/` and `Malware/` PCAP/7z files below `data/raw/ustc-tfc2016/`. Ten malware families for closed/open-set and prototype adaptation; benign captures are not used in the primary malware-family task. |
| ISCXVPN2016 | [UNB VPN-nonVPN dataset](https://www.unb.ca/cic/datasets/vpn.html) | Provider ZIPs `NonVPN-PCAPs-01.zip`, `NonVPN-PCAPs-02.zip`, `NonVPN-PCAPs-03.zip`, `VPN-PCAPS-01.zip`, `VPN-PCAPS-02.zip` below `data/raw/iscxvpn2016/`; four activity classes across VPN/non-VPN capture folds. |

A separate APT dataset tree exists in the original workspace, but no frozen APT experiment is represented here. It is excluded and no APT evaluation is claimed. ISCX email/P2P categories are excluded from the primary protocol because independent capture groups are insufficient. USTC provider licensing does not license this project's code; dataset terms remain separate.

## Release directory structure

The tree lists every included directory; individual files are enumerated in the tables below and in `RELEASE_VALIDATION.md`. There are no empty placeholder directories.

```text
reproducibility_release/
├── README.md
├── CITATION.cff
├── RELEASE_VALIDATION.md
├── paired_open_set_wilcoxon.py
├── paired_open_set_auroc.csv
├── config/                          # phase1.json: historical SDN configuration
├── controller/                      # forwarding and feature/inference Ryu applications
├── topology/                        # original four-host research topology
├── scripts/
│   ├── run_controller_phase4.sh     # shared Ryu launcher
│   ├── phase2/                      # dataset inspection, extraction and frozen partitions
│   ├── phase3/                      # model training/evaluation and result aggregation
│   ├── phase4/                      # inference service, synthetic benchmark and helpers
│   └── paper/                       # frozen audit, statistics inputs and table/figure builders
├── artifacts/
│   ├── phase1/                      # original controller lock and environment metadata
│   ├── phase2/                      # protocols, mappings, capture/support manifests, summaries
│   ├── phase3/
│   │   ├── cicids/
│   │   │   ├── final/               # repeated stratified baseline summaries
│   │   │   └── day_holdout/         # capture-day results
│   │   ├── iscx/
│   │   │   ├── sequence_models/     # architecture comparison summaries
│   │   │   └── early_sequence/      # common-cohort prefix comparison and IDs
│   │   └── ustc/
│   │       ├── closed_set/          # five seed reports, summaries, historical checksums
│   │       └── open_set/            # twenty method/rotation reports and few-shot summaries
│   ├── phase4/
│   │   └── benchmark/              # compact synthetic benchmark and GPU summaries
│   └── paper/
│       ├── mentor_comment3_4_final/ # matched-horizon, adaptation and compatibility tables
│       └── experimental_completion/# aggregate confusion matrices and model complexity
└── reproducibility/                # selection, original-state snapshots, checks and inventory
```

The phase-3 ML requirements lock is directly under `artifacts/phase3/`. Preserving paths avoids changing imports or hard-coded output locations. `data/`, `models/`, `.venv-*`, `logs/`, and caches are deliberately absent; future reproduction creates them inside a separate execution copy.

## Experimental methodology

**CICIDS2017:** the preparation scripts canonicalize column names and damaged Web Attack labels, remove invalid numerical rows, exclude conflicting feature/label groups, and remove repeated feature representations. The frozen cleaning report records 2,830,743 input rows, 2,867 invalid rows, 6,666 conflicting rows, 301,806 duplicate rows, and 2,519,404 retained rows. Two 64-bit pandas hashes provide a 128-bit feature fingerprint in the low-memory finalizer. Stable source-row identifiers and seed `20261011` determine the class-stratified 70/15/15 benchmark. Day holdout uses independent validation and test days; Monday is a benign-only false-positive challenge and is distinguished from binary attack days. The tree protocol records that Random Forest configuration was frozen after initial benchmark exposure; XGBoost pilot selection used validation data. This is a limitation of selection independence.

**Flow reconstruction:** USTC and ISCX extraction uses TShark metadata for IPv4/IPv6 TCP/UDP, canonical bidirectional endpoint pairs and 120-second idle segmentation. The first observed direction is positive, the reverse negative. Ethernet frame length (`frame.len`) supplies packet size; IATs are in seconds. Up to 50 observations are stored. Addresses and ports identify flows during extraction but are excluded from learned input, as are payload bytes and protocol identifiers. USTC requires at least three packets; primary ISCX requires at least five. PCAP-derived intermediate tables contain endpoint metadata and are excluded from redistribution.

**USTC partitioning:** SHA-256 fingerprints cover signed lengths and IATs rounded to microseconds. Conflicting cross-family representations are excluded; repeated within-family representations stay in the same partition. A temporal representation-group split yields 60/20/20 training/validation/test partitions. This is not independent-capture validation: most families have one source capture. The ten-family closed CNN–BiLSTM uses five stochastic seeds on the same frozen split, not five independent datasets.

**Normalization and training:** signed sizes use training-valid-token RMS scaling without centering. `log1p(IAT)` uses training-token mean and standard deviation. USTC sequences are right padded and use packed recurrent lengths. Open-set normalization uses known-family training flows only. Neural models use the AdamW configurations, class weighting or balanced sampling, validation stopping, and fixed seeds in the supplied JSON protocols. Model structures and hyperparameters remain unchanged.

**ISCX:** captures are assigned to five deterministic activity/domain-stratified folds. Run `r` tests fold `r`, validates fold `(r+1) mod 5`, and trains on the other three. The training-only cap is 500 flows per capture. Primary architecture comparison uses five packets. The separate early-sequence experiment retrains at N=5/10/20/50 on the same 1,811 flows with at least 50 packets; it must not be conflated with truncating frozen USTC models. The common-cohort identity file is included.

**Open set and calibration:** each of ten USTC rotations withholds one target family and a distinct calibration family, leaving eight known families. CE+MSP uses one minus maximum softmax probability. SupCon uses normalized embeddings, normalized training prototypes and cosine distance. Thresholds maximize known-vs-unknown balanced accuracy using known validation data and the separate calibration-family pool; the target family is excluded from threshold tuning. The frozen protocol records tie-breaking and all rotation identities. Between-rotation SD measures family/task variability, not stochastic-seed variability.

**Few shot:** frozen nested support IDs at K=1/5/10 construct an additional target-family prototype without gradient updates. Temporally later query samples remain distinct from the support pool. Nine-family F1 and target-family F1 are different outcomes. Heterogeneous support can worsen a single prototype; the study does not establish monotonic benefit from larger K.

**Frozen five-packet evaluation:** `scripts/paper/audit_comment3_4_final.py` compares original and five-packet horizons on matched eligible USTC cohorts using frozen normalization and checkpoints. Original-horizon results on the full test population are kept separate. The original threshold and an offline N=5 recalibration are reported separately; recalibration does not change the deployed historical operating point. This evaluator does not retrain the networks. It requires omitted checkpoints, model-ready data and, for original closed-set comparisons, saved predictions.

**Paired comparisons:** `paired_open_set_wilcoxon.py` pairs CE and SupCon unknown AUROC by target family, checks matching seed sets, averages within family, then performs a two-sided Wilcoxon signed-rank test across ten family rotations. It writes a per-family CSV and prints W/p; it does not treat samples from a single flow test set as independent rotations. The ISCX early summary also reports paired prefix differences by capture fold. No significance claim has been recomputed for this package.

**SDN:** Ryu/OpenFlow 1.3, Open vSwitch and Mininet connect a four-host/two-switch topology to an HTTP inference service on localhost. The monitor observes five packets at DPID 1, records monotonic controller IATs, and calls the separate ML service. The service selects only SupCon rotation 0 (Cridex target, Geodo calibration), with its frozen threshold. Synthetic TCP traffic measures processing, RTT, loss and throughput, not malware detection accuracy. Latency starts after the fifth observed packet; it excludes the time needed to collect five packets. Controller-observed IAT is not identical by definition to PCAP capture time. The existing final audit was `PARTIAL` because paired controller/PCAP packet-timing agreement was `NOT_VERIFIED`; packaging does not resolve that gap.

## Dependencies and verified environment

The lockfiles are copied observations of the research environments, not newly solved installation specifications or proof of compatibility on another system.

| Component | Verified evidence | Recorded environment |
|---|---|---|
| ML | Original `.venv-ml/pyvenv.cfg` inspected without copying private paths; `artifacts/phase3/ml_requirements_with_torch.lock.txt` | CPython 3.12; torch 2.13.0+cu132, NumPy 2.5.4, pandas 3.0.6, PyArrow 26.0.0, scikit-learn 1.9.1, XGBoost 3.4.1, SciPy 1.18.1, DuckDB 1.5.6, matplotlib 3.11.2, py7zr 1.1.3. |
| Controller | `artifacts/phase1/environment_manifest.json`, `controller_requirements.lock.txt`, `ryu_commit.txt`, `ryu_tag.txt` | Python 3.9.25; Ryu 4.34 at commit `c776e4cb68600b2ee0a4f38364f4a355502777f1`; eventlet 0.30.2, dnspython 1.16.0, six 1.16.0. |
| Testbed | `artifacts/phase1/environment_manifest.json` | Mininet 2.3.0, Open vSwitch 3.7.1, OpenFlow 1.3, iperf 2.2.1, TShark 4.6.4; Ubuntu 26.04 LTS on WSL2. System Python 3.14.4 is historical metadata, not the ML interpreter. |
| ISCX archive extraction | `artifacts/phase2/iscx_extraction_environment.txt` | 7zip package `26.00+dfsg-1`, `/usr/bin/7z`; Python zipfile could not decode the original ISCX compression method. |

The frozen neural training scripts explicitly require CUDA and use `cuda:0`; no CPU training mode is supplied. The frozen audit accepts `--device cpu`, `cuda`, or `auto`; the HTTP service accepts `--device`. This does not establish support for arbitrary Python, CUDA or framework versions. The original CUDA wheel index/build source is absent from the lockfile and must be supplied by the authors if the pinned `+cu132` wheel cannot be resolved. Mininet is a system dependency, not in the ML lock. Ryu's legacy build used pip 23.3.2/setuptools 57.5.0/wheel 0.37.1 in the original installer; this is a bootstrap prerequisite, not permission to run that excluded installer.

## Reproduction instructions (not executed during packaging)

### Workspace and output rules

Use an **independent standalone execution copy of this release**, outside the original research checkout, and preserve this packaged copy as evidence. All commands below assume the current directory is the standalone release root. Set `RELEASE_ROOT` in every terminal, and return to it before each command block. No command should use the parent research checkout as an input or output root. Do not create symlinks to original artifacts.

The scripts resolve their own release root with `__file__`, or use the current directory for relative outputs. Preprocessing may delete/replace staged tables and database files **inside that execution copy**. Training, summaries, plots and audits overwrite fixed artifact/report paths listed below. Shell batch launchers skip reports that already look valid, so they can skip packaged historical runs; direct training loops below are intentional. Historical checksum manifests are provenance, not manifests for newly trained outputs. Do not rewrite them to make frozen validation pass.

```bash
# Run from the root of your standalone execution copy.
export RELEASE_ROOT="$(pwd -P)"
test -f "$RELEASE_ROOT/README.md"
test -f "$RELEASE_ROOT/reproducibility/selection_plan.json"
cd "$RELEASE_ROOT"
mkdir -p .cache/pip .tmp
export PIP_CACHE_DIR="$RELEASE_ROOT/.cache/pip"
export TMPDIR="$RELEASE_ROOT/.tmp"
export PYTHONDONTWRITEBYTECODE=1
```

### 1. Environments and dependency installation

These are future setup commands, not packaging validation. Use preinstalled Python 3.12 and 3.9. Mininet, Open vSwitch, TShark, 7z, iperf, curl, Bash, GNU time and (for GPU sampling) nvidia-smi must already be available in the execution environment. No system installation script is included.

```bash
cd "$RELEASE_ROOT"
python3.12 -m venv .venv-ml
.venv-ml/bin/python -m pip install -r artifacts/phase3/ml_requirements_with_torch.lock.txt
python3.9 -m venv .venv-controller
.venv-controller/bin/python -m pip install pip==23.3.2 setuptools==57.5.0 wheel==0.37.1
.venv-controller/bin/python -m pip install eventlet==0.30.2 dnspython==1.16.0 six==1.16.0
.venv-controller/bin/python -m pip install --no-build-isolation -r artifacts/phase1/controller_requirements.lock.txt
```

The controller lock fetches the exact upstream Ryu commit through its editable Git requirement; third-party source is not bundled. All dependencies remain subject to availability and legacy-build compatibility. Installation has not been attempted. The ML installation may require the missing original wheel index; stop at that dependency gap instead of changing frozen framework versions and claiming equivalence.

### 2. Obtain and arrange official inputs

Use the official links in the dataset table; download into the directories below while preserving the original archive names. Acquisition may require provider forms or terms, so no guessed direct-download command is supplied. Copy only provider datasets into this standalone execution workspace.

```bash
cd "$RELEASE_ROOT"
mkdir -p data/raw/cicids2017 data/raw/ustc-tfc2016 data/raw/iscxvpn2016
mkdir -p data/interim/cicids2017/raw_csv
# After obtaining MachineLearningCSV.zip from UNB:
.venv-ml/bin/python -m zipfile -e data/raw/cicids2017/MachineLearningCSV.zip data/interim/cicids2017/raw_csv
```

Place USTC's `Benign/` and `Malware/` directly below its raw directory (not inside an extra provider-repository folder). Put the five ISCX ZIPs directly below its raw directory. The included capture manifests record names and CRC/hash identities where originally available; they contain no packet payloads.

### 3. Preprocess and reconstruct frozen splits

**Generates/replaces `data/interim`, `data/processed`, and `artifacts/phase2` outputs in the execution copy.**

```bash
cd "$RELEASE_ROOT"
.venv-ml/bin/python scripts/phase2/audit_raw_datasets.py
.venv-ml/bin/python scripts/phase2/audit_cicids2017.py
.venv-ml/bin/python scripts/phase2/prepare_cicids2017.py
.venv-ml/bin/python scripts/phase2/extract_ustc_tfc2016.py
.venv-ml/bin/python scripts/phase2/extract_ustc_flows.py
.venv-ml/bin/python scripts/phase2/freeze_ustc_protocol.py
.venv-ml/bin/python scripts/phase3/build_ustc_closed_model_ready.py
.venv-ml/bin/python scripts/phase2/audit_iscxvpn_archives.py
.venv-ml/bin/python scripts/phase2/freeze_iscxvpn_protocol.py
.venv-ml/bin/python scripts/phase2/extract_iscxvpn_flows.py
.venv-ml/bin/python scripts/phase2/freeze_iscxvpn_model_protocol.py
```

CICIDS `prepare_cicids2017.py` stages all eight CSVs before its database work; the original bounded-memory finalization is separately available:

```bash
# Only after the complete CICIDS stage/*.parquet and cicids_audit.json exist.
# Replaces CICIDS processed data and split summaries using the existing staged rows.
cd "$RELEASE_ROOT"
.venv-ml/bin/python scripts/phase2/finalize_cicids2017_lowmem.py
```

The included cleaning report describes the final low-memory artifact. The initial full preparation can be memory intensive; it has no staging-only flag. If it fails before all staging files exist, the low-memory finalizer is not an independent raw-data entry point. Neither pathway has been exercised during packaging. USTC protocol freezing regenerates the excluded representation manifest and nested support identities; ISCX freezing regenerates capture folds/model-ready data and its training cap. Compare regenerated counts/IDs to frozen metadata before interpreting downstream metrics.

### 4. Train and evaluate baselines and closed-set models

**Training also evaluates restored best-validation models and overwrites model/report/prediction/history paths.** There is no invented standalone `evaluate.py` or checkpoint flag.

```bash
cd "$RELEASE_ROOT"
for model in rf xgb; do
  for seed in 20261011 20261012 20261013 20261014 20261015; do
    .venv-ml/bin/python scripts/phase3/run_cicids_tree_final.py --model "$model" --seed "$seed" --threads 2
  done
done
bash scripts/phase3/run_all_cicids_day_holdout.sh
for seed in 20261011 20261012 20261013 20261014 20261015; do
  .venv-ml/bin/python scripts/phase3/run_ustc_closed_set.py --seed "$seed"
done
for model in cnn lstm bilstm cnn_bilstm; do
  for run in 0 1 2 3 4; do
    .venv-ml/bin/python scripts/phase3/run_iscx_sequence_model.py --model "$model" --run "$run"
  done
done
```

Day-holdout per-run JSON reports are intentionally excluded, so its batch launcher can run all five days × five seeds × two models in a fresh execution copy. Rerunning it later skips valid reports. The corresponding direct entry point is `run_cicids_day_holdout.py --model rf --run 0 --seed 20261011 --threads 2`. USTC closed-set CUDA training and test evaluation are in the same seed command.

### 5. Open-set training, calibration and few-shot evaluation

**Writes `models/phase3/ustc/open_set/` and `artifacts/phase3/ustc/open_set/`.** Each SupCon run also evaluates K=1/5/10 prototype adaptation, with no separate adaptation-training command.

```bash
cd "$RELEASE_ROOT"
for method in ce_msp supcon_proto; do
  for run in 0 1 2 3 4 5 6 7 8 9; do
    .venv-ml/bin/python scripts/phase3/run_ustc_open_set.py --method "$method" --run "$run"
  done
done
.venv-ml/bin/python scripts/phase3/summarize_ustc_open_set.py
```

The included original reports allow inspection and paired statistical analysis without the model weights. Replacing those reports through training produces a new experiment, not the original frozen checkpoint.

### 6. Five-packet and other early observations

ISCX common-cohort **retraining** at each prefix, followed by paired fold summaries:

```bash
cd "$RELEASE_ROOT"
for n in 5 10 20 50; do
  for run in 0 1 2 3 4; do
    .venv-ml/bin/python scripts/phase3/run_iscx_early_sequence.py --sequence-length "$n" --run "$run"
  done
done
.venv-ml/bin/python scripts/phase3/summarize_iscx_early_sequence.py
```

USTC **frozen-model inference** uses the following conditional command. **It is blocked in the lightweight package until the original checkpoints, model-ready Parquet, saved closed-set predictions, and matching historical manifests/artifacts are independently obtained in the standalone execution workspace.** Retraining does not guarantee the original SHA-256 values. This audit writes/replaces `artifacts/paper/mentor_comment3_4_final/` and hashes existing artifacts; it may be expensive even without training.

```bash
# Original frozen artifacts must first be restored at the documented relative paths.
cd "$RELEASE_ROOT"
.venv-ml/bin/python scripts/paper/audit_comment3_4_final.py --root "$RELEASE_ROOT" --device cpu --batch-size 512 --threads 4
```

That same audit checks per-family few-shot results, nested supports, threshold recalibration, and deployment rotation 0. It does not establish controller/PCAP IAT agreement in the absence of paired packet observations.

### 7. Aggregation and paired statistical analysis

**Summary scripts overwrite their corresponding aggregate CSVs.** CICIDS/ISCX summaries require the per-run reports generated by their training commands (omitted to reduce package size). USTC summaries can consume included reports.

```bash
cd "$RELEASE_ROOT"
.venv-ml/bin/python scripts/phase3/summarize_cicids_tree_final.py
.venv-ml/bin/python scripts/phase3/summarize_cicids_day_holdout.py
.venv-ml/bin/python scripts/phase3/summarize_iscx_sequence_models.py
.venv-ml/bin/python scripts/phase3/summarize_iscx_early_sequence.py
.venv-ml/bin/python scripts/phase3/summarize_ustc_closed_set.py
.venv-ml/bin/python scripts/phase3/summarize_ustc_open_set.py
.venv-ml/bin/python paired_open_set_wilcoxon.py artifacts/phase3/ustc/open_set --output paired_open_set_auroc.csv
```

The statistical command works from the twenty included USTC reports once SciPy is installed. It overwrites `paired_open_set_auroc.csv` and prints the Wilcoxon result. `scripts/paper/build_closed_set_confusion_matrix.py` uses the five closed reports to write aggregate confusion tables and a figure. `compute_model_complexity.py` additionally requires the omitted SupCon rotation-0 checkpoint. `build_results_package.py` uses **hard-coded frozen numerical arrays** to write tables and figures; it is a presentation builder, not an independent metric recomputation. `audit_review2_experiments.py` checks report fields and external `sha256sum` manifests; binary/data exclusions mean a complete historical acceptance check cannot pass from this package alone.

### 8. SDN integration and synthetic overhead benchmark

**Conditional on original SupCon rotation-0 checkpoint matching `open_set_sha256.txt`, its included original report, and installed system testbed dependencies.** The inference service checks checkpoint/report SHA-256; a newly trained replacement is not automatically accepted. No permissive replacement hashes are supplied.

Run the service from a separate terminal at the standalone root:

```bash
cd "$RELEASE_ROOT"
.venv-ml/bin/python scripts/phase4/run_inference_service.py --host 127.0.0.1 --port 8765 --device cuda:0
```

For each mode `forwarding`, `features`, `inference`, start the following controller command in a controller terminal and stop it before switching modes:

```bash
cd "$RELEASE_ROOT"
bash scripts/phase4/run_controller_benchmark_mode.sh forwarding
# Later, repeat with features, then inference; service must be healthy for inference.
```

Run the matching benchmark from a testbed terminal with Mininet available to `/usr/bin/python3`:

```bash
cd "$RELEASE_ROOT"
sudo /usr/bin/python3 scripts/phase4/run_phase4_benchmark.py --mode forwarding --trials 5 --microflows 500
# Repeat for features and inference with the corresponding controller active.
```

Mininet needs network-administration privileges and creates transient virtual switches/interfaces. These future testbed operations were not performed during packaging. They write mode benchmark JSON under `artifacts/phase4/benchmark/`. The controller wrapper **truncates** mode telemetry on start and overwrites `logs/controller.log`; repeat only in the execution copy. `tcp_microflow_client.py`/`tcp_microflow_server.py` are launched by the benchmark, using the local `.venv-ml` interpreter.

After all modes have produced telemetry:

```bash
cd "$RELEASE_ROOT"
.venv-ml/bin/python scripts/phase4/summarize_phase4_benchmark.py
```

This summary requires at least 500 feature/inference records; the original JSONL telemetry is excluded. `sample_gpu_memory.sh PID OUTPUT` samples per-process GPU usage and truncates the caller-specified output. It does **not** reconstruct the original incremental whole-device GPU baseline by itself; the original `gpu_memory_summary.json` is included as evidence. No complete baseline/whole-device GPU measurement orchestrator was found. The general `topology/research_topology.py --mode interactive` entry point is separate from the phase-4 benchmark topology construction.

## Research results and populations

Numbers below are read from included original tables, with displayed rounding only. They were **not regenerated during packaging**, and are not independently checked against an unavailable manuscript. `artifacts/paper/master_results.csv` is an earlier rounded consolidation; the final matched-horizon audit tables supply later cohort-specific values. Keep their populations and precision separate.

| Population / experiment | Result | Included source |
|---|---|---|
| CICIDS stratified split, five seeds | RF macro-F1 0.998078; XGBoost 0.998888 | `artifacts/paper/master_results.csv`; `artifacts/phase3/cicids/final/tree_final_summary.csv` |
| CICIDS capture-day holdout | Equal-day macro-F1 RF 0.529178; XGBoost 0.500492 | `artifacts/paper/master_results.csv`; `artifacts/phase3/cicids/day_holdout/day_holdout_binary_summary.csv` |
| ISCX primary five-packet architecture comparison, five capture folds | CNN–BiLSTM macro-F1 0.375958 ± 0.078866 | `artifacts/phase3/iscx/sequence_models/sequence_model_summary.csv` |
| ISCX common ≥50-packet cohort, retrained N=20 | Macro-F1 0.467906 ± 0.131010 | `artifacts/phase3/iscx/early_sequence/early_sequence_summary.csv` |
| USTC original full closed test, original up-to-50 horizon, five seeds | Macro-F1 approximately 0.738051 ± 0.012824 | `artifacts/paper/mentor_comment3_4_final/closed_five_packet_summary.csv` |
| USTC matched ≥5-packet closed test, frozen models | Original-horizon macro-F1 0.648897 ± 0.025216 → N=5 macro-F1 0.223454 ± 0.067423 | Same final closed summary; distinct from original full test |
| USTC ten-family rotations, original full test/query horizon | CE+MSP unknown AUROC approximately 0.625537 ± 0.105720; SupCon approximately 0.652948 ± 0.191855 (original reports) | `artifacts/phase3/ustc/open_set/zero_shot_summary.csv` |
| USTC ten rotations, matched ≥5-packet test/query, N=5 | CE AUROC 0.418092 ± 0.222739; SupCon 0.497523 ± 0.252246; open macro-F1 0.138781 and 0.211927 respectively | `artifacts/paper/mentor_comment3_4_final/open_five_packet_summary.csv` |
| USTC few shot, ten rotations, original horizon K=5 | Target-family F1 approximately 0.439168 ± 0.301118; nine-family macro-F1 0.674043 ± 0.063215 | `artifacts/phase3/ustc/open_set/few_shot_summary.csv` |
| Single deployed SupCon rotation 0, five-packet offline Cridex cohort | AUROC 0.883849; original threshold rejects all known samples (known false rejection 1.0), open macro-F1 0.080120 | `artifacts/paper/mentor_comment3_4_final/deployment_run0_five_packet.csv`, original operating point |
| Same single checkpoint, **offline N=5 recalibration**, not original deployment | Known false rejection 0.228704; open macro-F1 0.530608; AUROC unchanged | Same deployment table, `offline_n5_recalibrated` row |
| Synthetic Mininet overhead | Mean controller decision 3.810099 ms; p95 5.636088 ms; p99 6.671421 ms after fifth packet; mean model forward 1.485059 ms | `artifacts/phase4/benchmark/phase4_pipeline_summary.csv` |
| Synthetic throughput / memory | Forwarding 94.76 Mbit/s vs inference 94.64 Mbit/s; zero measured loss; original incremental whole-device GPU summary 233 MiB | `artifacts/phase4/benchmark/phase4_benchmark_summary.json`, `gpu_memory_summary.json` |

The strong single Cridex AUROC does not represent the ten-family average. Threshold calibration changes an operating point, not score-ranking AUROC. SDs across closed seeds, ISCX folds and open-set tasks describe different variation. Synthetic packet loss and throughput are not measures of unseen-malware robustness.

## Excluded artifacts and reproducibility limits

| Original relative path / pattern | Purpose | Exclusion and consequence |
|---|---|---|
| `data/raw/{cicids2017,ustc-tfc2016,iscxvpn2016}/` | Original provider archives/captures | Dataset redistribution excluded; obtain independently from providers. |
| `data/raw/APT/` | Separate raw materials | Outside frozen experiments represented here; no claims made. |
| `data/interim/`, `data/processed/` | PCAPs, staged flow tables, DuckDB databases, model-ready ISCX/CICIDS data | Binary/large/data artifacts; reconstruct only in the execution workspace. |
| `artifacts/phase2/ustc_malware_protocol_manifest.parquet` | Per-flow representation-aware split assignments | Binary excluded; regenerate with `freeze_ustc_protocol.py`. |
| `artifacts/phase2/iscx_model_manifest.parquet`, `cicids_split_assignments.parquet` | Per-flow folds/splits | Binary excluded; regeneration needed. Compact protocols are included. |
| `artifacts/phase3/ustc/closed_set/ustc_closed_model_ready.parquet` | USTC model-ready sequences | Binary excluded; build script included. Exact frozen audit needs original matching bytes. |
| `models/phase3/ustc/closed_set/cnn_bilstm_seed_*.pt` | Five frozen closed checkpoints | All weights excluded; exact frozen inference unavailable from package alone. |
| `models/phase3/ustc/open_set/{ce_msp,supcon_proto}_run_*.pt` | Open-set encoders/prototypes/normalization | All weights excluded; training reconstructs new models, not guaranteed original bytes. |
| `models/phase3/ustc/open_set/supcon_proto_run_0.pt` | Single original deployed checkpoint | Required by inference service/hash gate and complexity script; no public checkpoint URL was found. Request the exact artifact from authors. |
| `models/phase3/{cicids,iscx}/`, other `models/` | Tree/sequence models (`.joblib`, `.ubj`, `.pt`) | Binary model formats excluded even if small. |
| `artifacts/phase3/**/*predictions.parquet` | Per-flow predictions and original closed audit reference | Binary/data excluded; original frozen evaluator needs saved closed predictions, not only JSON reports. |
| CICIDS/ISCX per-run `*_report.json` and training histories | Detailed intermediate repetition records | Nonessential in the lightweight package; regenerate to reaggregate those results. USTC reports retained for paired statistics and confusion matrices. |
| `artifacts/phase4/**/*.jsonl`, `logs/`, `*.log` | Per-flow telemetry and execution logs | Excluded; phase-4 aggregation needs new telemetry. Original controller/PCAP timing agreement remains unverified. |
| `.venv-*`, `vendor/`, `vendor-osken/`, Git internals, private environment/configuration trees | Installed environments and third-party material | Excluded; locks record dependencies, no credentials or machine-home configurations copied. |
| `config/datasets.json` | Unused dataset overview | Original contains two concatenated JSON objects (`Extra data`, line 47); excluded unchanged. Protocol JSONs provide authoritative parameters. |
| Original installation/cleanup scripts, pilots, demos and root audit shim | Historical setup and exploratory workflows | Unneeded for selected dependency chain; original controller installer deletes/recreates environments and changes vendor checkout, so it is not bundled or used. |
| Manuscript/literature PDFs, images, binaries, caches and temporary files | Non-code or presentation material | Not included; no manuscript was available for metadata verification. |

No file over 1 MiB was copied. Excluded formats include `.pt`, `.pth`, `.ckpt`, `.joblib`, `.pkl`, `.pickle`, `.onnx`, `.h5`, `.hdf5`, `.safetensors`, `.npy`, `.npz`, `.parquet`, `.arrow`, `.feather`, `.pcap`, `.pcapng`, `.bin`, archives, databases, plus observed `.ubj` and `.duckdb`. No substitutes were generated.

Hardware, CUDA kernels, library versions, capture clocks, OS networking and workload affect runtime and potentially near-tie predictions. Five closed seeds share one split; ten open rotations use one seed each, and limited captures constrain external validity. Original full-horizon performance is not five-packet deployment performance. Exact frozen-checkpoint inference requires the original artifacts and checksums; retraining under documented protocols is a separate reproducibility path. No exact runtime or numerical reproduction is claimed.

## File-by-file implementation and configuration reference

Every included implementation, protocol, configuration and original evidence file is listed below. Inputs and outputs are relative to the release root unless the table explicitly names the local service or system testbed. Included evidence files do not run by themselves; “none (evidence)” means no output is generated by reading them. Reproduction output paths refer to a future standalone execution copy.

| Relative file path | Purpose | Inputs | Outputs |
|---|---|---|---|
| `controller/l2_learning.py` | Reference L2 learning switch; companion inspected by frozen feature audit | Ryu OpenFlow 1.3 events | Forwarding FlowMod/PacketOut messages |
| `controller/phase4_monitor.py` | L2 forwarding with feature extraction, five-packet HTTP inference and telemetry | OpenFlow events; PHASE4_* environment; optional localhost inference service | FlowMod/PacketOut; PHASE4_METRICS_PATH JSONL (wrapper supplies release-local path) |
| `paired_open_set_wilcoxon.py` | Pair AUROCs by target/seed; two-sided Wilcoxon test across ten target families | results_directory with CE/SupCon JSON reports; SciPy | --output CSV (default paired_open_set_auroc.csv); W/p and paired summary printed |
| `scripts/paper/audit_comment3_4_final.py` | Frozen original/N=5 audit, threshold comparison, support nesting and feature-code audit | Original USTC checkpoints/data/reports/predictions/checksums; phase2/3/4 sources; --root | artifacts/paper/mentor_comment3_4_final/ tables, audit log, JSON and original hashes; no training |
| `scripts/paper/audit_review2_experiments.py` | Check experiment acceptance fields and historical SHA-256 manifests | Closed/open reports, confusion/complexity summaries, SDN/GPU summaries and original manifests | artifacts/paper/experimental_completion/review2_experiment_acceptance.csv; stdout; missing excluded files prevent full acceptance |
| `scripts/paper/build_closed_set_confusion_matrix.py` | Aggregate and plot normalized confusion matrices across closed seeds | Five original closed *_report.json files | artifacts/paper/experimental_completion/closed_set_confusion_{mean,std,sources}.csv and PNG |
| `scripts/paper/build_results_package.py` | Write paper tables/plots from hard-coded frozen arrays | Embedded metric/plot arrays in this file | artifacts/paper/master_results.csv, claims_matrix.csv, results_digest.md and figures; not a fresh experiment |
| `scripts/paper/compute_model_complexity.py` | Measure encoder parameter/operation counts and checkpoint size | SupCon rotation-0 checkpoint; open protocol; run_ustc_open_set module | artifacts/paper/experimental_completion/model_complexity.{csv,json} |
| `scripts/phase2/audit_cicids2017.py` | Inspect eight CSV schemas, labels and counts | data/interim/cicids2017/raw_csv/ | artifacts/phase2/cicids_audit.json, cicids_file_summary.csv, cicids_class_distribution.csv |
| `scripts/phase2/audit_iscxvpn_archives.py` | Inventory ZIP capture members and activity/domain labels | data/raw/iscxvpn2016/*.zip | artifacts/phase2/iscxvpn_capture_manifest.{csv,json}, iscxvpn_category_summary.csv |
| `scripts/phase2/audit_raw_datasets.py` | Inventory and hash independently obtained raw dataset files | data/raw/ | artifacts/phase2/raw_dataset_inventory.{csv,json} |
| `scripts/phase2/extract_iscxvpn_flows.py` | Extract selected captures with 7z; reconstruct packet sequences with inherited folds | ISCX ZIPs and iscxvpn_protocol_manifest.csv | data/processed/iscxvpn2016/flows/; artifacts/phase2/iscx_flow_* summaries, statistics and logs |
| `scripts/phase2/extract_ustc_flows.py` | Reconstruct bidirectional idle-delimited packet sequences via TShark | USTC interim PCAPs and ustc_capture_manifest.csv | data/processed/ustc-tfc2016/flows/; artifacts/phase2/ustc_flow_* summaries/protocol (dry-run uses separate paths) |
| `scripts/phase2/extract_ustc_tfc2016.py` | Extract provider PCAP and 7z files into class directories | data/raw/ustc-tfc2016/Benign/ and Malware/ | data/interim/ustc-tfc2016/pcap/; artifacts/phase2/ustc_capture_manifest.csv |
| `scripts/phase2/finalize_cicids2017_lowmem.py` | Bounded-memory finalization using dual feature hashes and the frozen split seed | Complete data/interim/cicids2017/stage/; artifacts/phase2/cicids_audit.json | stage_hashed/, cicids2017_lowmem.duckdb, data/processed/cicids2017/cicids2017_clean.parquet; artifacts/phase2/cicids_split_assignments.parquet and summaries |
| `scripts/phase2/freeze_iscxvpn_model_protocol.py` | Construct model-ready sequences and freeze eligibility/capture cap | ISCX flow Parquets; artifacts/phase2/iscx_flow_capture_summary.csv | data/processed/iscxvpn2016/iscx_model_ready.parquet; artifacts/phase2/iscx_model_manifest.parquet, iscx_model_protocol.json and summary CSVs |
| `scripts/phase2/freeze_iscxvpn_protocol.py` | Freeze capture-group activity/domain folds | artifacts/phase2/iscxvpn_capture_manifest.csv | artifacts/phase2/iscxvpn_protocol_manifest.csv, iscxvpn_fold_summary.csv, iscxvpn_cv_protocol.json |
| `scripts/phase2/freeze_ustc_protocol.py` | Freeze representation groups, temporal splits, rotations and nested support IDs | USTC malware flow Parquets | artifacts/phase2/ustc_malware_protocol_manifest.parquet, ustc_protocol.json, ustc_fewshot_support.csv, ustc_malware_protocol_summary.csv |
| `scripts/phase2/prepare_cicids2017.py` | Stage/clean flow CSVs, remove conflicts/duplicates, construct deterministic splits | Eight raw CSVs | data/interim/cicids2017/stage/, data/processed/cicids2017/cicids2017_clean.parquet; artifacts/phase2 mappings/splits/summaries |
| `scripts/phase3/build_ustc_closed_model_ready.py` | Join flow representations to the frozen malware manifest | USTC flow Parquets and ustc_malware_protocol_manifest.parquet | artifacts/phase3/ustc/closed_set/ustc_closed_model_ready.parquet |
| `scripts/phase3/run_all_cicids_day_holdout.sh` | Batch the frozen runs; skip reports passing embedded validation | scripts/phase3/run_cicids_day_holdout.py; .venv-ml/bin/python; existing report checks | Child model/results plus logs/resources at artifacts/phase3/cicids/day_holdout/ |
| `scripts/phase3/run_all_iscx_early_sequence.sh` | Batch the frozen runs; skip reports passing embedded validation | scripts/phase3/run_iscx_early_sequence.py; .venv-ml/bin/python; existing report checks | Child model/results plus logs/resources at artifacts/phase3/iscx/early_sequence/ |
| `scripts/phase3/run_all_iscx_sequence_models.sh` | Batch the frozen runs; skip reports passing embedded validation | scripts/phase3/run_iscx_sequence_model.py; .venv-ml/bin/python; existing report checks | Child model/results plus logs/resources at artifacts/phase3/iscx/sequence_models/ |
| `scripts/phase3/run_all_ustc_closed_set.sh` | Batch the frozen runs; skip reports passing embedded validation | scripts/phase3/run_ustc_closed_set.py; .venv-ml/bin/python; existing report checks | Child model/results plus logs/resources at artifacts/phase3/ustc/closed_set/ |
| `scripts/phase3/run_all_ustc_open_set.sh` | Batch the frozen runs; skip reports passing embedded validation | scripts/phase3/run_ustc_open_set.py; .venv-ml/bin/python; existing report checks | Child model/results plus logs/resources at artifacts/phase3/ustc/open_set/ |
| `scripts/phase3/run_cicids_day_holdout.py` | Train/evaluate RF/XGBoost on one day rotation and seed | CICIDS clean Parquet; column/day mappings and tree/day protocols | models/phase3/cicids/day_holdout/; artifacts/phase3/cicids/day_holdout/*_report.json |
| `scripts/phase3/run_cicids_tree_final.py` | Train RF/XGBoost and evaluate binary stratified baseline | CICIDS clean Parquet; column mapping; tree_final_protocol.json | models/phase3/cicids/final/; artifacts/phase3/cicids/final/*_report.json and predictions |
| `scripts/phase3/run_iscx_early_sequence.py` | Train/evaluate each prefix on common ≥50-packet cohort | ISCX model-ready Parquet; phase2/early protocols and common cohort IDs | models/phase3/iscx/early_sequence/; artifacts/phase3/iscx/early_sequence/ reports, predictions, histories |
| `scripts/phase3/run_iscx_sequence_model.py` | Train/evaluate CNN, LSTM, BiLSTM or CNN–BiLSTM on capture folds | ISCX model-ready Parquet; phase2 and sequence_model protocols | models/phase3/iscx/sequence_models/; artifacts/phase3/iscx/sequence_models/ reports, predictions, histories |
| `scripts/phase3/run_ustc_closed_set.py` | Define CNN–BiLSTM; train/evaluate ten-family closed model | USTC model-ready Parquet; closed_set_protocol.json; --seed | models/phase3/ustc/closed_set/; artifacts/phase3/ustc/closed_set/ reports, predictions, histories |
| `scripts/phase3/run_ustc_open_set.py` | Define encoders/prototypes; train, independently calibrate, evaluate open/few-shot tasks | USTC model-ready Parquet; phase2/open protocols and nested supports; --method/--run | models/phase3/ustc/open_set/; artifacts/phase3/ustc/open_set/ method/rotation reports and histories |
| `scripts/phase3/summarize_cicids_day_holdout.py` | Aggregate holdout days/seeds; separate benign-only day | artifacts/phase3/cicids/day_holdout/*_report.json; relevant frozen protocol | artifacts/phase3/cicids/day_holdout/day_holdout_runs.csv, day_holdout_day_summary.csv, day_holdout_binary_summary.csv |
| `scripts/phase3/summarize_cicids_tree_final.py` | Aggregate repeated stratified baseline reports | artifacts/phase3/cicids/final/*_report.json; relevant frozen protocol | artifacts/phase3/cicids/final/tree_final_runs.csv, tree_final_summary.csv |
| `scripts/phase3/summarize_iscx_early_sequence.py` | Aggregate prefixes and paired capture-fold deltas | artifacts/phase3/iscx/early_sequence/*_report.json; relevant frozen protocol | artifacts/phase3/iscx/early_sequence/early_sequence_{runs,summary,paired_deltas,domain_summary,class_summary}.csv |
| `scripts/phase3/summarize_iscx_sequence_models.py` | Aggregate architecture, domain and class results | artifacts/phase3/iscx/sequence_models/*_report.json; relevant frozen protocol | artifacts/phase3/iscx/sequence_models/sequence_model_{runs,summary,domain_summary,class_summary}.csv |
| `scripts/phase3/summarize_ustc_closed_set.py` | Aggregate seed and per-family closed metrics | artifacts/phase3/ustc/closed_set/*_report.json; relevant frozen protocol | artifacts/phase3/ustc/closed_set/closed_set_{runs,summary,class_summary}.csv |
| `scripts/phase3/summarize_ustc_open_set.py` | Aggregate open, paired-target and few-shot results | artifacts/phase3/ustc/open_set/*_report.json; relevant frozen protocol | artifacts/phase3/ustc/open_set/zero_shot_{runs,summary,paired_by_target,paired_summary}.csv; few_shot_{runs,summary}.csv |
| `scripts/phase4/run_controller_benchmark_mode.sh` | Set five-packet monitor mode and launch Ryu; truncate selected telemetry | forwarding/features/inference argument; health endpoint for inference; scripts/run_controller_phase4.sh | artifacts/phase4/benchmark/{mode}_controller_metrics.jsonl; logs/controller.log |
| `scripts/phase4/run_inference_service.py` | Serve original rotation-0 SupCon inference over localhost HTTP; verify frozen hashes | models/phase3/ustc/open_set/supcon_proto_run_0.pt; matching report; open protocol/hash manifest; run_ustc_open_set module | HTTP /health and /infer responses on port 8765; process stdout |
| `scripts/phase4/run_phase4_benchmark.py` | Construct Mininet topology; measure synthetic network and microflow overhead | Active matching controller mode; system Mininet; .venv-ml; TCP helper scripts | artifacts/phase4/benchmark/{mode}_benchmark.json |
| `scripts/phase4/sample_gpu_memory.sh` | Sample process GPU memory with nvidia-smi 100 times | PID and caller-specified OUTPUT inside execution workspace | Truncates/writes OUTPUT; not the incremental whole-device summary builder |
| `scripts/phase4/summarize_phase4_benchmark.py` | Aggregate network trials and ≥500 feature/inference records | Three mode benchmark JSONs and controller_metrics.jsonl (telemetry excluded) | artifacts/phase4/benchmark/phase4_benchmark_summary.json, phase4_pipeline_summary.csv |
| `scripts/phase4/tcp_microflow_client.py` | Generate repeated synthetic request/response connections | Host/port, connection count and request/response sizes | JSON microflow timing summary to stdout |
| `scripts/phase4/tcp_microflow_server.py` | Accept finite synthetic TCP request/response connections | Host/port, connection count and request/response sizes | Socket responses and stdout status |
| `scripts/run_controller_phase4.sh` | Launch Ryu phase-4 monitor with OpenFlow TCP port 6653 | .venv-controller/bin/ryu-manager; controller/phase4_monitor.py | logs/controller.log overwritten via tee; active controller |
| `topology/research_topology.py` | Original four-host/two-switch Mininet smoke/interactive topology | System Mininet; external controller at localhost:6653; --mode | Transient network, smoke metrics or interactive CLI/stdout |

### Configuration, manifests, dependency records and evidence files

| Relative file path | Purpose | Inputs | Outputs |
|---|---|---|---|
| `artifacts/paper/claims_matrix.csv` | Original evidence/claim wording and limitation matrix | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/experimental_completion/closed_set_confusion_mean.csv` | Aggregate closed-family confusion mean | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/experimental_completion/closed_set_confusion_sources.csv` | Aggregate closed-family confusion sources | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/experimental_completion/closed_set_confusion_std.csv` | Aggregate closed-family confusion sample standard deviation | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/experimental_completion/model_complexity.csv` | Frozen encoder parameter/operation/checkpoint-size summary | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/experimental_completion/model_complexity.json` | Frozen encoder parameter/operation/checkpoint-size summary | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/experimental_completion/review2_experiment_acceptance.csv` | Original experiment acceptance checklist evidence; not packaging validation | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/master_results.csv` | Earlier rounded cross-experiment consolidation; contains separate metric populations | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/closed_five_packet_per_family.csv` | Matched versus full-cohort horizon evidence: closed_five_packet_per_family.csv; read horizon/cohort/metric fields separately | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/closed_five_packet_per_seed.csv` | Matched versus full-cohort horizon evidence: closed_five_packet_per_seed.csv; read horizon/cohort/metric fields separately | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/closed_five_packet_summary.csv` | Matched versus full-cohort horizon evidence: closed_five_packet_summary.csv; read horizon/cohort/metric fields separately | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/closed_original_reproduction.csv` | Existing audit comparison with original frozen reports; packaging has not rerun it | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/deployment_run0_five_packet.csv` | Single deployed checkpoint: original and offline recalibrated N=5 operating points | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/feature_compatibility_audit.csv` | Source-based offline/controller representation compatibility evidence; not paired packet-timing validation | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/fewshot_manuscript_table.md` | Original few-shot adaptation fewshot_manuscript_table.md | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/fewshot_summary_verified.csv` | Original few-shot adaptation fewshot_summary_verified.csv | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/fewshot_support_nesting.csv` | Original nested support agreement fewshot_support_nesting.csv | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/fewshot_target_family_f1.csv` | Original per-family adaptation fewshot_target_family_f1.csv | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/five_packet_threshold_comparison.csv` | Per-method/rotation N=5 original versus offline recalibrated threshold comparison | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/open_five_packet_per_family.csv` | Matched versus full-cohort horizon evidence: open_five_packet_per_family.csv; read horizon/cohort/metric fields separately | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/open_five_packet_per_rotation.csv` | Matched versus full-cohort horizon evidence: open_five_packet_per_rotation.csv; read horizon/cohort/metric fields separately | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/open_five_packet_summary.csv` | Matched versus full-cohort horizon evidence: open_five_packet_summary.csv; read horizon/cohort/metric fields separately | Original experiment/provenance record | None (evidence) |
| `artifacts/paper/mentor_comment3_4_final/open_original_reproduction.csv` | Existing audit comparison with original frozen reports; packaging has not rerun it | Original experiment/provenance record | None (evidence) |
| `artifacts/phase1/controller_requirements.lock.txt` | Original pinned controller/Ryu environment freeze | Original pip freeze | Installation input; no installation executed |
| `artifacts/phase1/environment_manifest.json` | Original OS, hardware, Python/controller/testbed versions and topology | Original experiment/provenance record | None (evidence) |
| `artifacts/phase1/ryu_commit.txt` | Exact upstream Ryu source commit | Original experiment/provenance record | None (evidence) |
| `artifacts/phase1/ryu_tag.txt` | Exact upstream Ryu source tag | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/cicids_audit.json` | Original raw CSV schema and class/count audit used by low-memory finalizer | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/cicids_benchmark_split_summary.csv` | Frozen aggregate/counts/paired evidence: cicids_benchmark_split_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/cicids_binary_distribution.csv` | Frozen aggregate/counts/paired evidence: cicids_binary_distribution.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/cicids_clean_class_distribution.csv` | Frozen aggregate/counts/paired evidence: cicids_clean_class_distribution.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/cicids_cleaning_report.json` | Frozen cleaning totals, dual hashes and deterministic split definitions | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/cicids_column_mapping.json` | Original and canonical feature-column order used by tree scripts | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/cicids_day_distribution.csv` | Frozen aggregate/counts/paired evidence: cicids_day_distribution.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/cicids_day_holdout_protocol.json` | Exact train/validation/test day sets and benign-only Monday scope | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/cicids_label_mapping.json` | Canonical ASCII class-name mappings including damaged Web Attack labels | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/iscx_extraction_environment.txt` | Original 7z package/executable and extractor hash provenance | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/iscx_flow_fold_summary.csv` | Frozen aggregate/counts/paired evidence: iscx_flow_fold_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/iscx_flow_protocol.json` | ISCX extraction units, eligible activities and inherited capture folds | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/iscx_model_cv_summary.csv` | Frozen aggregate/counts/paired evidence: iscx_model_cv_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/iscx_model_protocol.json` | Model input, five rotations, eligibility and training-only capture cap | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/iscxvpn_capture_manifest.csv` | Provider ZIP member names, activity/domain, byte sizes and CRC identities | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/iscxvpn_cv_protocol.json` | Five capture-group folds, strata and excluded activities | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/iscxvpn_protocol_manifest.csv` | Capture-member activity/domain labels, inclusion decisions and deterministic folds | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/ustc_capture_manifest.csv` | Provider/class capture names, source-relative paths, sizes and SHA-256 identities | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/ustc_fewshot_support.csv` | Frozen nested support flow IDs by target family and K; no packets or endpoint addresses | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/ustc_final_sha256.txt` | Unchanged historical SHA-256 manifest; also names excluded artifacts; not a complete release checksum list | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/ustc_flow_protocol.json` | USTC extraction units, segmentation and sequence definitions | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase2/ustc_malware_protocol_summary.csv` | Frozen aggregate/counts/paired evidence: ustc_malware_protocol_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase2/ustc_protocol.json` | Representation grouping, temporal splits, ten target/calibration rotations, nested supports | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase3/cicids/day_holdout/day_holdout_binary_summary.csv` | Frozen aggregate/counts/paired evidence: day_holdout_binary_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/cicids/day_holdout/day_holdout_day_summary.csv` | Frozen aggregate/counts/paired evidence: day_holdout_day_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/cicids/day_holdout/day_holdout_runs.csv` | Frozen aggregate/counts/paired evidence: day_holdout_runs.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/cicids/final/tree_final_runs.csv` | Frozen aggregate/counts/paired evidence: tree_final_runs.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/cicids/final/tree_final_summary.csv` | Frozen aggregate/counts/paired evidence: tree_final_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/cicids/tree_final_protocol.json` | Frozen RF/XGBoost hyperparameters, thresholds, five seeds and metrics | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase3/iscx/early_sequence/early_sequence_class_summary.csv` | Frozen aggregate/counts/paired evidence: early_sequence_class_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/early_sequence/early_sequence_domain_summary.csv` | Frozen aggregate/counts/paired evidence: early_sequence_domain_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/early_sequence/early_sequence_paired_deltas.csv` | Frozen aggregate/counts/paired evidence: early_sequence_paired_deltas.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/early_sequence/early_sequence_runs.csv` | Frozen aggregate/counts/paired evidence: early_sequence_runs.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/early_sequence/early_sequence_summary.csv` | Frozen aggregate/counts/paired evidence: early_sequence_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/early_sequence/ge50_common_cohort_flow_ids.txt` | Exact flow identity list for the ISCX common ≥50-packet cohort | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/early_sequence_protocol.json` | Common cohort, N=5/10/20/50, split counts and paired-fold protocol | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase3/iscx/sequence_model_protocol.json` | Four neural architectures, primary N=5 inputs and training settings | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase3/iscx/sequence_models/sequence_model_class_summary.csv` | Frozen aggregate/counts/paired evidence: sequence_model_class_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/sequence_models/sequence_model_domain_summary.csv` | Frozen aggregate/counts/paired evidence: sequence_model_domain_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/sequence_models/sequence_model_runs.csv` | Frozen aggregate/counts/paired evidence: sequence_model_runs.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/iscx/sequence_models/sequence_model_summary.csv` | Frozen aggregate/counts/paired evidence: sequence_model_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ml_requirements_with_torch.lock.txt` | Original pinned ML/CUDA environment freeze | Original pip freeze | Installation input; no installation executed |
| `artifacts/phase3/ustc/closed_set/closed_set_class_summary.csv` | Frozen aggregate/counts/paired evidence: closed_set_class_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/closed_set_runs.csv` | Frozen aggregate/counts/paired evidence: closed_set_runs.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/closed_set_sha256.txt` | Unchanged historical SHA-256 manifest; also names excluded artifacts; not a complete release checksum list | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/closed_set_summary.csv` | Frozen aggregate/counts/paired evidence: closed_set_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261011_report.json` | Frozen closed-set seed report (metrics, confusion matrix, timing, model reference) | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261012_report.json` | Frozen closed-set seed report (metrics, confusion matrix, timing, model reference) | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261013_report.json` | Frozen closed-set seed report (metrics, confusion matrix, timing, model reference) | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261014_report.json` | Frozen closed-set seed report (metrics, confusion matrix, timing, model reference) | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261015_report.json` | Frozen closed-set seed report (metrics, confusion matrix, timing, model reference) | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set/model_ready_sha256.txt` | Unchanged historical SHA-256 manifest; also names excluded artifacts; not a complete release checksum list | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/closed_set_protocol.json` | Ten families, frozen 60/20/20 counts, CNN–BiLSTM and five-seed training | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_0_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_1_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_2_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_3_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_4_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_5_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_6_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_7_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_8_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/ce_msp_run_9_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/few_shot_runs.csv` | Original few-shot adaptation few_shot_runs.csv | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/few_shot_summary.csv` | Original few-shot adaptation few_shot_summary.csv | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/open_set_sha256.txt` | Unchanged historical SHA-256 manifest; also names excluded artifacts; not a complete release checksum list | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_0_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_1_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_2_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_3_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_4_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_5_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_6_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_7_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_8_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/supcon_proto_run_9_report.json` | Frozen open-set method/target report, threshold, normalization and SupCon few-shot metrics where applicable | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/zero_shot_paired_by_target.csv` | Frozen aggregate/counts/paired evidence: zero_shot_paired_by_target.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/zero_shot_paired_summary.csv` | Frozen aggregate/counts/paired evidence: zero_shot_paired_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/zero_shot_runs.csv` | Frozen aggregate/counts/paired evidence: zero_shot_runs.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set/zero_shot_summary.csv` | Frozen aggregate/counts/paired evidence: zero_shot_summary.csv; field names and precision retained exactly | Original experiment/provenance record | None (evidence) |
| `artifacts/phase3/ustc/open_set_protocol.json` | CE/SupCon training, independent calibration, unknown scoring and K=1/5/10 adaptation | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase4/benchmark/features_benchmark.json` | Synthetic features network trial and microflow timing summary | Original experiment/provenance record | None (evidence) |
| `artifacts/phase4/benchmark/forwarding_benchmark.json` | Synthetic forwarding network trial and microflow timing summary | Original experiment/provenance record | None (evidence) |
| `artifacts/phase4/benchmark/gpu_memory_summary.json` | Original incremental whole-device GPU memory evidence; baseline capture not reconstructed here | Original experiment/provenance record | None (evidence) |
| `artifacts/phase4/benchmark/inference_benchmark.json` | Synthetic inference network trial and microflow timing summary | Original experiment/provenance record | None (evidence) |
| `artifacts/phase4/benchmark/phase4_benchmark_summary.json` | Synthetic SDN network/pipeline latency, throughput/loss and overhead aggregate | Original experiment/provenance record | None (evidence) |
| `artifacts/phase4/benchmark/phase4_pipeline_summary.csv` | Synthetic controller and service timing distributions in microseconds | Original experiment/provenance record | None (evidence) |
| `artifacts/phase4/benchmark_protocol.json` | Five network trials, 500 microflows, latency metrics and synthetic-accuracy policy | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `artifacts/phase4/phase4_protocol.json` | SDN topology, feature units, five-packet horizon and exact deployed model/report mapping | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `config/phase1.json` | Historical four-host/two-switch controller and OpenFlow configuration | Read by corresponding implementation or retained as frozen metadata | None (configuration/evidence; never altered during packaging) |
| `paired_open_set_auroc.csv` | Original ten-family CE/SupCon paired AUROC table | Original experiment/provenance record | None (evidence) |

### Packaging records

| Relative file path | Purpose | Inputs | Outputs |
|---|---|---|---|
| `README.md` | Workflow, file reference, scientific limitations, access and attribution | Inspected source/protocols and author-supplied metadata | Documentation |
| `CITATION.cff` | Citation metadata without an invented DOI, email, release date or license | Author-provided manuscript title/names/affiliations | Citation metadata |
| `RELEASE_VALIDATION.md` | Final file list, counts, exceptions and non-destructive checks | Static validation and integrity records | Packaging report |
| `reproducibility/selection_plan.json` | Explicit 164-file copy allowlist and excluded category/extension counts | Inspected original filesystem | Selection provenance |
| `reproducibility/original_state_before.json` | Pre-copy Git state, selected SHA-256 and research/Git file metadata | Read-only original research tree; Git with optional locks disabled | Before snapshot |
| `reproducibility/source_copy_integrity.json` | Original/copy hashes, comment-removal counts and preserved-comment exceptions | Copied source/evidence | Copy provenance |
| `reproducibility/implementation_inventory.json` | Per-script descriptions, inputs/outputs, AST imports and exact argument declarations | Static source parsing only | Machine-readable interface inventory |
| `reproducibility/validation_results.json` | Syntax, file, interface, privacy-pattern and documentation checks | Release files; no research script imports/execution | Static validation record |
| `reproducibility/original_state_after.json` | Final Git/source recheck and research/Git metadata differences | Before snapshot and read-only originals | After comparison; reports actual scope |
| `reproducibility/release_file_inventory.csv` | Every final file and byte size, including this inventory and validation report | Completed release | Inventory; sizes stabilized after generation |
| `reproducibility/release_sha256.txt` | SHA-256 of every release file except this checksum file itself | Final release bytes | Release integrity manifest; historical manifests remain separate |

Seven Python files are preserved byte-for-byte because included historical manifests hash them; shell comments are retained because Bash heredocs/expansion/directives require stronger proof than syntax checking. Other Python comment removal replaces only nonessential COMMENT tokens with spaces, preserving line positions, all statements/docstrings, shebangs, encodings and directives. ASTs including type comments and non-comment executable token streams are compared before/after. Exact per-file exceptions and hashes appear in `source_copy_integrity.json`.

## Citation, licensing and attribution

Use `CITATION.cff` to cite this code package and identify the associated manuscript by the title/authors above. Publication status, year, publisher, repository URL, version and DOI have not been verified and are intentionally absent. Add the actual GitHub/Zenodo identifiers only after an authorized release and archive exist; nothing was uploaded or pushed during packaging.

No applicable project-wide license was found. **Author confirmation of a software license is required before public redistribution.** No `LICENSE` was generated and no license grant is inferred. The MPL-2.0 file under the original USTC provider dataset and the licenses in excluded Ryu/OS-Ken vendor trees apply to those third-party materials, not automatically to this project. Existing copyright/license notices in copied source are preserved; third-party packages retain their own terms. Cite and attribute each dataset according to its provider's instructions. This local package can be reviewed now; public-release readiness still depends on licensing and metadata confirmation.

## Research Reproducibility Release Disclaimer

This repository is a code-focused reproducibility release prepared in support of the manuscript's journal submission and subsequent academic research release. It contains selected implementation files, experiment configurations, evaluation procedures, and shareable research summaries. Original third-party datasets, trained model checkpoints, and large intermediate artifacts are intentionally excluded. The release is provided for research verification and reproducibility purposes and does not constitute a production-ready malware detection or network security system.
