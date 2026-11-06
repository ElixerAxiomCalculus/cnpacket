# Release validation report

**Source-code packaging completed. Static validation passed with the documented metadata and reproduction limitations below. Full scientific reproduction was not executed.**

- Original repository root: `/home/sayak/CNPacket`.
- Release directory: `/home/sayak/CNPacket/reproducibility_release`.
- Packaging date: 6 November 2026, Asia/Calcutta (UTC snapshot times recorded in JSON).
- Files copied from original: **164**; generated packaging/documentation files: **11**; total: **175**.
- Total release file bytes: **2,321,788 bytes** (2.214 MiB). This sums regular-file lengths; filesystem allocation and directory entries are excluded.
- Copied file bytes: **1,274,483**. No copied file exceeds 1 MiB; no size exception was used.
- Original filesystem files excluded: **46,865**, counted across the inspected tree, including installed environments/vendor/Git trees. Counts are inventory facts, not claims of excluded experimental runs. `selection_plan.json` supplies full root/extension counts.

## Directory-wise file counts

Counts below are direct files in each directory, not recursively double-counted totals. Every release directory contains a file or an included descendant; none is an empty placeholder.

| Directory | Files |
|---|---:|
| `.` | 5 |
| `artifacts/paper` | 2 |
| `artifacts/paper/experimental_completion` | 6 |
| `artifacts/paper/mentor_comment3_4_final` | 15 |
| `artifacts/phase1` | 4 |
| `artifacts/phase2` | 23 |
| `artifacts/phase3` | 1 |
| `artifacts/phase3/cicids` | 1 |
| `artifacts/phase3/cicids/day_holdout` | 3 |
| `artifacts/phase3/cicids/final` | 2 |
| `artifacts/phase3/iscx` | 2 |
| `artifacts/phase3/iscx/early_sequence` | 6 |
| `artifacts/phase3/iscx/sequence_models` | 4 |
| `artifacts/phase3/ustc` | 2 |
| `artifacts/phase3/ustc/closed_set` | 10 |
| `artifacts/phase3/ustc/open_set` | 27 |
| `artifacts/phase4` | 2 |
| `artifacts/phase4/benchmark` | 6 |
| `config` | 1 |
| `controller` | 2 |
| `reproducibility` | 8 |
| `scripts` | 1 |
| `scripts/paper` | 5 |
| `scripts/phase2` | 11 |
| `scripts/phase3` | 18 |
| `scripts/phase4` | 7 |
| `topology` | 1 |

## File types included

| Extension | Files |
|---|---:|
| `.cff` | 1 |
| `.csv` | 58 |
| `.json` | 56 |
| `.md` | 3 |
| `.py` | 38 |
| `.sh` | 8 |
| `.txt` | 11 |

## Excluded file categories and types

The following mutually exclusive inventory categories use precedence (for example, binary PCAPs under `data/` are counted as datasets). All prohibited formats are excluded regardless of category.

| Category | Files |
|---|---:|
| Git internals or third-party vendor source | 5,128 |
| cache or compiled file | 42 |
| dataset or dataset-side material; not redistributed | 816 |
| image or manuscript/presentation binary | 5 |
| logs or per-flow telemetry | 360 |
| not selected: nonessential implementation, logs, summaries or private/system material | 536 |
| prohibited binary/large artifact format | 199 |
| virtual environment | 39,779 |

Excluded checkpoints/models include `.pt`, `.joblib`, `.ubj`; observed data/artifacts include `.pcap`, `.parquet`, `.duckdb`, ZIP and 7z archives. The broader prohibited-format list in README also covers `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.onnx`, HDF5, safetensors, NumPy arrays, Arrow/Feather, PCAPNG, binaries, archives, database/SQLite formats, even where absent. Raw/processed datasets are excluded regardless of text extension or size. Images, compiled caches, environments, vendor sources, logs, telemetry and private configuration are omitted. No binary/checkpoint exception was permitted.

No whole directory was copied. Every copied file was checked individually for existence, regular-file status, UTF-8 text, absence of NULs, extension eligibility, ≤1 MiB size, JSON validity where applicable and sensitive-content patterns. No symlinks or hardlinks were created; independent regular-file copies preserve original relative paths. Files requiring companion modules retain them in `scripts/phase3/`. No credentials, raw endpoint tables, registration numbers or payloads were intentionally included. Pattern scanning supplements the explicit selection and content review; it is not a universal secret-detection guarantee. The original root path and Git inventory are retained only in the explicitly requested integrity records.

## Source-to-copy and scientific integrity

All **164** original selected-file SHA-256 values were recorded before copying and rechecked afterward: **unchanged**. Original/copied hashes and exceptions are in `reproducibility/source_copy_integrity.json`. All non-code evidence/configuration/requirements/results copies are byte-identical. The only comment-processed file is `scripts/paper/audit_comment3_4_final.py` (five COMMENT tokens replaced by equal-length spaces); executable tokens, AST including type comments and compiled bytecode match the original. Function docstrings, shebangs, encoding lines, license/copyright and recognized tool directives were preserved.

Seven Python files were deliberately retained byte-for-byte because they appear in included frozen checksum manifests:

- `scripts/phase2/extract_ustc_flows.py`.
- `scripts/phase2/freeze_ustc_protocol.py`.
- `scripts/phase3/build_ustc_closed_model_ready.py`.
- `scripts/phase3/run_ustc_closed_set.py`.
- `scripts/phase3/run_ustc_open_set.py`.
- `scripts/phase3/summarize_ustc_closed_set.py`.
- `scripts/phase3/summarize_ustc_open_set.py`.

All eight Bash files retain comments: embedded Python heredocs, shell expansion and possible external directives mean that comment removal was not proven safe. `bash -n` provides a syntax check, not a proof that arbitrary shell-comment rewriting preserves behavior. Python files with no eligible comments remain byte-identical.

No executable logic, validation statements, random seeds, split policies, hyperparameters, loss/metric/statistical definitions, threshold-selection rules, result values or historical manifests were changed. No fabricated data/checkpoints/results were created. All 38 Python files compiled to identical bytecode when original and copy were compiled with the same virtual filename; compilation did not execute them or write bytecode caches.

Historical manifests: **50** included-file entries match their original expected SHA-256 values; **113** entries refer to deliberately omitted artifacts. Historical manifests remain unchanged and are **not complete-release verification lists**. Running a blanket historical `sha256sum -c` will therefore report absent artifacts. The new `reproducibility/release_sha256.txt` covers all current release files except itself.

## Static syntax and structural validation

| Check | Outcome | Scope / limitation |
|---|---|---|
| Python syntax | PASS: 38 files | `ast.parse` with type comments; no project module imported/executed |
| Python code equivalence | PASS: 38 files | AST, non-comment token comparison during processing, compiled bytecode comparison |
| Shell syntax | PASS: 8 files | `bash -n`; no shell script executed |
| Copied JSON | PASS: 50 files | Standard JSON parser; frozen protocol values unchanged |
| Copied CSV | PASS: 57 files | Parsed records and consistent column count; original values/bytes preserved |
| CITATION.cff | Core checks PASS | YAML parsed; required metadata/type/four authors checked; full CFF schema validator not run |
| File formats / size / links | PASS | UTF-8 text, no NULs, prohibited extensions, symlinks or >1 MiB files |
| Imported local helper modules | PASS | `run_ustc_closed_set` and `run_ustc_open_set` present; dynamic imports/relative script references inspected |
| Documentation source paths | PASS: 46 implementation references | All exist at documented paths |
| Documented Python CLI flags | PASS | Compared to AST `add_argument` declarations; no `--help` execution |
| Sensitive-content checks | PASS for reviewed patterns | Private-key/known-token and selected-source absolute-home patterns, allowlist/content review; no external secret scanner installed |
| Runtime dependencies / scientific results | NOT EXECUTED | Neither installations nor experiments/inference/statistical reruns were performed |

Dependencies identified from static imports and original locks include NumPy, pandas, PyArrow, DuckDB, py7zr, torch, scikit-learn, XGBoost, joblib, SciPy, matplotlib, Ryu and system Mininet. TShark, 7z, Open vSwitch, iperf, curl, Bash, GNU time and nvidia-smi support the documented external workflows. Presence in a lockfile does not prove availability or compatibility on the recipient's machine.

## Documentation completeness

README documents the author-supplied manuscript title, four authors/affiliations, corresponding author, all three actual frozen datasets and official sources, every implementation/configuration/evidence file, all actual release folders, environmental provenance, checked command interfaces, outputs that may be overwritten, population-specific metrics, excluded artifacts and limitations. It ends with the required Research Reproducibility Release Disclaimer. File descriptions, exact parser declarations and imports also appear in `implementation_inventory.json`.

All four ORCIDs are explicitly **“Not provided — to be completed before release”**. No manuscript was available to verify the proposed correspondence email, so no email was included. The author-supplied title cannot be independently confirmed as the latest manuscript title from this checkout. CITATION.cff contains no invented DOI, release date, software version, publisher, repository URL, email or ORCID.

No applicable project-wide software license was found; no LICENSE was generated. Dataset/vendor licenses were not adopted for project code. **Author licensing confirmation is required before public redistribution.** The task did not authorize selecting a license or uploading/publishing anything, and neither was done.

## Missing requirements, blockers and warnings

- **Datasets:** provider raw inputs are required. All processed flow/feature tables and binary split manifests are excluded; reconstruct them only inside a standalone execution copy.
- **Exact frozen inference:** closed/open checkpoints, model-ready USTC Parquet, saved closed predictions and matching historical artifacts must be obtained from authors. No checkpoint download URL was found. Retraining can produce new models under the original protocol but cannot guarantee frozen hashes.
- **Service checksum gate:** the SDN inference service verifies the original rotation-0 checkpoint/report against `open_set_sha256.txt`. A retrained checkpoint is not a drop-in replacement. Hashes/gates were not weakened or rewritten.
- **CUDA/dependencies:** neural training explicitly requires CUDA. Python 3.12 ML and Python 3.9.25 Ryu environments are recorded; no general support matrix was established. The torch `2.13.0+cu132` wheel source/index is not in the original lock. Legacy Ryu bootstrap compatibility and dependency installation remain untested. System Mininet must be independently provisioned.
- **Aggregation inputs:** CICIDS/ISCX per-run JSON reports, training histories, predictions and SDN JSONL telemetry are excluded. Their aggregate evidence is included; rerunning those summarizers requires regenerated inputs. Twenty original USTC open reports and five closed reports are included for paired statistics and confusion summaries.
- **GPU baseline:** process sampling helper is included, but no complete incremental whole-device baseline measurement orchestrator was found. The original GPU summary is retained as evidence.
- **Original configuration issue:** `config/datasets.json` contains concatenated JSON (`Extra data`, line 47); it is unused by selected scripts and excluded unchanged. No repair/refactoring was performed.
- **Original scientific audit:** existing `final_audit_summary.json` was PARTIAL, with controller/PCAP packet-timing agreement NOT_VERIFIED and no recorded errors. This large machine-specific summary is excluded; compact final tables remain. No new packet-timing evidence was fabricated.
- **Presentation builder:** `build_results_package.py` contains hard-coded frozen metric arrays. Its output is presentation evidence, not an independent experimental reproduction.
- **Metadata/licensing:** title/email/ORCID verification and software licensing require author completion before public release. No publication or Zenodo DOI was invented.
- **Validation scope:** static checks do not demonstrate dependency installation, GPU compatibility, statistical significance, model performance or SDN runtime reproducibility. Full CFF schema validation is outstanding; YAML/core checks passed.

## Original repository integrity

Before copying, Git status, tracked diff fingerprint, all selected source hashes and metadata for **3,725** research/Git files were recorded in `original_state_before.json`. After copying/documentation, `original_state_after.json` records:

- Selected source hash changes: **0 of 164**.
- Research/Git file metadata changes (size, nanosecond mtime, mode/link target): **0 of 3,725**.
- Pre-existing Git status and tracked diff fingerprint: **unchanged**, excluding only new untracked `reproducibility_release/` entries.
- Pre-existing tracked modification: `artifacts/phase1/smoke_test.log`; preserved.
- Pre-existing untracked files: **1,173**; preserved.

No original was restored, cleaned, renamed, moved, deleted, refactored or executed. Git calls used optional locks disabled during the before/after validation. All file writes were restricted to this new release directory. Datasets/checkpoints were not reread for full content hashing; artifact integrity evidence is unchanged metadata plus write confinement, and selected-file content hashes. Installed environment/vendor/private trees were counted but excluded from content snapshots. Root-directory mtime and access times are intentionally outside comparison because adding a directory and reading originals changes them.

## Exact included file paths

The complete list includes copied files and generated documentation/provenance. `release_file_inventory.csv` records final sizes and origins. `release_sha256.txt` excludes only itself from content hashing to avoid a circular checksum.

```text
CITATION.cff
README.md
RELEASE_VALIDATION.md
artifacts/paper/claims_matrix.csv
artifacts/paper/experimental_completion/closed_set_confusion_mean.csv
artifacts/paper/experimental_completion/closed_set_confusion_sources.csv
artifacts/paper/experimental_completion/closed_set_confusion_std.csv
artifacts/paper/experimental_completion/model_complexity.csv
artifacts/paper/experimental_completion/model_complexity.json
artifacts/paper/experimental_completion/review2_experiment_acceptance.csv
artifacts/paper/master_results.csv
artifacts/paper/mentor_comment3_4_final/closed_five_packet_per_family.csv
artifacts/paper/mentor_comment3_4_final/closed_five_packet_per_seed.csv
artifacts/paper/mentor_comment3_4_final/closed_five_packet_summary.csv
artifacts/paper/mentor_comment3_4_final/closed_original_reproduction.csv
artifacts/paper/mentor_comment3_4_final/deployment_run0_five_packet.csv
artifacts/paper/mentor_comment3_4_final/feature_compatibility_audit.csv
artifacts/paper/mentor_comment3_4_final/fewshot_manuscript_table.md
artifacts/paper/mentor_comment3_4_final/fewshot_summary_verified.csv
artifacts/paper/mentor_comment3_4_final/fewshot_support_nesting.csv
artifacts/paper/mentor_comment3_4_final/fewshot_target_family_f1.csv
artifacts/paper/mentor_comment3_4_final/five_packet_threshold_comparison.csv
artifacts/paper/mentor_comment3_4_final/open_five_packet_per_family.csv
artifacts/paper/mentor_comment3_4_final/open_five_packet_per_rotation.csv
artifacts/paper/mentor_comment3_4_final/open_five_packet_summary.csv
artifacts/paper/mentor_comment3_4_final/open_original_reproduction.csv
artifacts/phase1/controller_requirements.lock.txt
artifacts/phase1/environment_manifest.json
artifacts/phase1/ryu_commit.txt
artifacts/phase1/ryu_tag.txt
artifacts/phase2/cicids_audit.json
artifacts/phase2/cicids_benchmark_split_summary.csv
artifacts/phase2/cicids_binary_distribution.csv
artifacts/phase2/cicids_clean_class_distribution.csv
artifacts/phase2/cicids_cleaning_report.json
artifacts/phase2/cicids_column_mapping.json
artifacts/phase2/cicids_day_distribution.csv
artifacts/phase2/cicids_day_holdout_protocol.json
artifacts/phase2/cicids_label_mapping.json
artifacts/phase2/iscx_extraction_environment.txt
artifacts/phase2/iscx_flow_fold_summary.csv
artifacts/phase2/iscx_flow_protocol.json
artifacts/phase2/iscx_model_cv_summary.csv
artifacts/phase2/iscx_model_protocol.json
artifacts/phase2/iscxvpn_capture_manifest.csv
artifacts/phase2/iscxvpn_cv_protocol.json
artifacts/phase2/iscxvpn_protocol_manifest.csv
artifacts/phase2/ustc_capture_manifest.csv
artifacts/phase2/ustc_fewshot_support.csv
artifacts/phase2/ustc_final_sha256.txt
artifacts/phase2/ustc_flow_protocol.json
artifacts/phase2/ustc_malware_protocol_summary.csv
artifacts/phase2/ustc_protocol.json
artifacts/phase3/cicids/day_holdout/day_holdout_binary_summary.csv
artifacts/phase3/cicids/day_holdout/day_holdout_day_summary.csv
artifacts/phase3/cicids/day_holdout/day_holdout_runs.csv
artifacts/phase3/cicids/final/tree_final_runs.csv
artifacts/phase3/cicids/final/tree_final_summary.csv
artifacts/phase3/cicids/tree_final_protocol.json
artifacts/phase3/iscx/early_sequence/early_sequence_class_summary.csv
artifacts/phase3/iscx/early_sequence/early_sequence_domain_summary.csv
artifacts/phase3/iscx/early_sequence/early_sequence_paired_deltas.csv
artifacts/phase3/iscx/early_sequence/early_sequence_runs.csv
artifacts/phase3/iscx/early_sequence/early_sequence_summary.csv
artifacts/phase3/iscx/early_sequence/ge50_common_cohort_flow_ids.txt
artifacts/phase3/iscx/early_sequence_protocol.json
artifacts/phase3/iscx/sequence_model_protocol.json
artifacts/phase3/iscx/sequence_models/sequence_model_class_summary.csv
artifacts/phase3/iscx/sequence_models/sequence_model_domain_summary.csv
artifacts/phase3/iscx/sequence_models/sequence_model_runs.csv
artifacts/phase3/iscx/sequence_models/sequence_model_summary.csv
artifacts/phase3/ml_requirements_with_torch.lock.txt
artifacts/phase3/ustc/closed_set/closed_set_class_summary.csv
artifacts/phase3/ustc/closed_set/closed_set_runs.csv
artifacts/phase3/ustc/closed_set/closed_set_sha256.txt
artifacts/phase3/ustc/closed_set/closed_set_summary.csv
artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261011_report.json
artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261012_report.json
artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261013_report.json
artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261014_report.json
artifacts/phase3/ustc/closed_set/cnn_bilstm_seed_20261015_report.json
artifacts/phase3/ustc/closed_set/model_ready_sha256.txt
artifacts/phase3/ustc/closed_set_protocol.json
artifacts/phase3/ustc/open_set/ce_msp_run_0_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_1_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_2_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_3_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_4_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_5_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_6_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_7_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_8_report.json
artifacts/phase3/ustc/open_set/ce_msp_run_9_report.json
artifacts/phase3/ustc/open_set/few_shot_runs.csv
artifacts/phase3/ustc/open_set/few_shot_summary.csv
artifacts/phase3/ustc/open_set/open_set_sha256.txt
artifacts/phase3/ustc/open_set/supcon_proto_run_0_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_1_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_2_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_3_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_4_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_5_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_6_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_7_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_8_report.json
artifacts/phase3/ustc/open_set/supcon_proto_run_9_report.json
artifacts/phase3/ustc/open_set/zero_shot_paired_by_target.csv
artifacts/phase3/ustc/open_set/zero_shot_paired_summary.csv
artifacts/phase3/ustc/open_set/zero_shot_runs.csv
artifacts/phase3/ustc/open_set/zero_shot_summary.csv
artifacts/phase3/ustc/open_set_protocol.json
artifacts/phase4/benchmark/features_benchmark.json
artifacts/phase4/benchmark/forwarding_benchmark.json
artifacts/phase4/benchmark/gpu_memory_summary.json
artifacts/phase4/benchmark/inference_benchmark.json
artifacts/phase4/benchmark/phase4_benchmark_summary.json
artifacts/phase4/benchmark/phase4_pipeline_summary.csv
artifacts/phase4/benchmark_protocol.json
artifacts/phase4/phase4_protocol.json
config/phase1.json
controller/l2_learning.py
controller/phase4_monitor.py
paired_open_set_auroc.csv
paired_open_set_wilcoxon.py
reproducibility/implementation_inventory.json
reproducibility/original_state_after.json
reproducibility/original_state_before.json
reproducibility/release_file_inventory.csv
reproducibility/release_sha256.txt
reproducibility/selection_plan.json
reproducibility/source_copy_integrity.json
reproducibility/validation_results.json
scripts/paper/audit_comment3_4_final.py
scripts/paper/audit_review2_experiments.py
scripts/paper/build_closed_set_confusion_matrix.py
scripts/paper/build_results_package.py
scripts/paper/compute_model_complexity.py
scripts/phase2/audit_cicids2017.py
scripts/phase2/audit_iscxvpn_archives.py
scripts/phase2/audit_raw_datasets.py
scripts/phase2/extract_iscxvpn_flows.py
scripts/phase2/extract_ustc_flows.py
scripts/phase2/extract_ustc_tfc2016.py
scripts/phase2/finalize_cicids2017_lowmem.py
scripts/phase2/freeze_iscxvpn_model_protocol.py
scripts/phase2/freeze_iscxvpn_protocol.py
scripts/phase2/freeze_ustc_protocol.py
scripts/phase2/prepare_cicids2017.py
scripts/phase3/build_ustc_closed_model_ready.py
scripts/phase3/run_all_cicids_day_holdout.sh
scripts/phase3/run_all_iscx_early_sequence.sh
scripts/phase3/run_all_iscx_sequence_models.sh
scripts/phase3/run_all_ustc_closed_set.sh
scripts/phase3/run_all_ustc_open_set.sh
scripts/phase3/run_cicids_day_holdout.py
scripts/phase3/run_cicids_tree_final.py
scripts/phase3/run_iscx_early_sequence.py
scripts/phase3/run_iscx_sequence_model.py
scripts/phase3/run_ustc_closed_set.py
scripts/phase3/run_ustc_open_set.py
scripts/phase3/summarize_cicids_day_holdout.py
scripts/phase3/summarize_cicids_tree_final.py
scripts/phase3/summarize_iscx_early_sequence.py
scripts/phase3/summarize_iscx_sequence_models.py
scripts/phase3/summarize_ustc_closed_set.py
scripts/phase3/summarize_ustc_open_set.py
scripts/phase4/run_controller_benchmark_mode.sh
scripts/phase4/run_inference_service.py
scripts/phase4/run_phase4_benchmark.py
scripts/phase4/sample_gpu_memory.sh
scripts/phase4/summarize_phase4_benchmark.py
scripts/phase4/tcp_microflow_client.py
scripts/phase4/tcp_microflow_server.py
scripts/run_controller_phase4.sh
topology/research_topology.py
```
