# Phase 1 manual label alignment audit

Date: 2026-11-07T10:46:13.743336+05:30. Reviewer: assistant, using tool-assisted record-by-record inspection; no external human sign-off is claimed.

**Verified: 25/25. Unresolved: 0/25. Mismatched: 0/25.** Mismatched means an observed numerical/identity discrepancy and is a subset of unresolved, not a third disjoint status.

## Sources and verification

Read `artifacts/modern_validation/ecu_malnett_v2/apt_v2_followup_20261107/label_review_samples.csv` and the original Construct sessions CSVs/PCAPs in `data/raw/APT/`: 24 captures, 27 directional Construct rows, 349 reviewed packets. Each listed CSV record number is the 1-based logical CSV record including header record 1; it is not the Construct `flow_N` identifier. No cached extracted sequences or prior matching verdicts were used as packet evidence.

A separate standard-library decoder read every original Ethernet/IPv4 PCAP frame and selected both directions of each exact IP/port/TCP-or-UDP tuple. Fresh TShark exact-tuple display filters cross-checked every selected frame number, stored timestamp, frame-relative timestamp, captured/wire length and tuple. CSV directional counts and the supplied bidirectional counts were compared to all observed packets. Original positive-row numbers, scores/reasons, canonical flow-ID derivation and frozen-manifest identity/count/split were checked without regenerating or changing splits. Full commands and per-flow findings are recorded in `audit_evidence.json`.

**Timestamp interpretation:** Construct `timestamp` is checked against the final directional packet's stored PCAP timestamp. `session_duration` is checked against last minus first. All captures use microsecond PCAP timestamps close to origin zero; these are not independently established wall-clock dates. TShark `frame.time_relative` subtracts the capture's first-frame timestamp (10–16 microseconds here), so blindly comparing that field to Construct `timestamp` would introduce an offset. No fitted per-flow offset was applied. The prespecified numerical tolerance is one microsecond, matching the PCAP timestamp resolution. Actual residuals are retained in `directional_evidence.csv`.

**Protocol interpretation:** TCP/UDP is verified from IP protocol and transport ports. Construct `protocol` is an application/dissector summary for TLS, QUIC or DNS in three rows; those are corroborated with packet dissection. Generic TCP summaries do not require every packet's application display label to be TCP. The two bidirectional samples (7 and 12) are checked as two separate directional Construct rows whose counts sum to the full observed flow.

## Record outcomes

| Sample | Family | Flow ID | Observed packets | Result |
|---:|---|---|---:|---|
| 1 | APT28 | `009a2323ca5391a1abafe1f6` | 37 | checked |
| 2 | APT28 | `2a4aa8160be8979994ec1132` | 28 | checked |
| 3 | APT28 | `00a9f6e884b05bf51a426c35` | 54 | checked |
| 4 | APT28 | `01467f85752b9cdb60aafef4` | 7 | checked |
| 5 | APT28 | `01ef7ed1f73d9fac70778279` | 5 | checked |
| 6 | APT29 | `0b5577dd313fbe6754b0a236` | 5 | checked |
| 7 | APT29 | `0aaa53d81afa683d4660b7a5` | 9 | checked |
| 8 | APT29 | `01f069b43a58b6da2a7f7eb5` | 7 | checked |
| 9 | APT29 | `12d49ee3214521dd1909f828` | 7 | checked |
| 10 | APT29 | `0d9e22ef0990171a40b4dc9a` | 7 | checked |
| 11 | Lazarus | `028ff88f72785e031ad4b8cc` | 9 | checked |
| 12 | Lazarus | `236930d751142670ea847551` | 8 | checked |
| 13 | Lazarus | `240dac684dd04169bfc69a4f` | 7 | checked |
| 14 | Lazarus | `0859719a187a14dc64eeca0a` | 12 | checked |
| 15 | Lazarus | `0058c562df520ee2897f363a` | 5 | checked |
| 16 | RustyStealer | `1918444836681a3df8fa3b9f` | 5 | checked |
| 17 | RustyStealer | `028e821bee6d9857334bfec5` | 7 | checked |
| 18 | RustyStealer | `28167ad8e243d1f1d43b3b94` | 8 | checked |
| 19 | RustyStealer | `01fce3295eaaec1afec37d79` | 18 | checked |
| 20 | RustyStealer | `11621cf1593563bcb056af36` | 29 | checked |
| 21 | Turla | `2ba92ed05dafa60dd76f7b5c` | 5 | checked |
| 22 | Turla | `11e6646c0756316da992d789` | 7 | checked |
| 23 | Turla | `0275a7c42929932a2d2283ae` | 32 | checked |
| 24 | Turla | `0bbf7c66c498853bff432257` | 12 | checked |
| 25 | Turla | `090662b9c3f15e5b69f44367` | 19 | checked |

## Outputs and limits

- `label_review_samples_verified.csv`: all 25 original rows and columns; only `review_status`/`review_notes` are populated in this independent copy.
- `directional_evidence.csv`: original CSV references, exact directional endpoints/protocols/counts, first/last timestamps, timing residuals and packet frame numbers.
- `packet_evidence.csv`: independent per-packet tuple/timing evidence and TShark protocol labels; no payload bytes are exported.
- `audit_evidence.json`: record-level decisions, parser checks, commands and capture diagnostics.
- `source_integrity.json`: before/after SHA-256 checks and original-file/Git comparisons.
- `verify_label_alignment.py`: new read-only audit implementation; it refuses to overwrite audit outputs.

Original integrity: 52 relevant input/source files were SHA-256 checked before/after; 2092 original data/source/model/result file metadata entries were unchanged; pre-existing Git status and tracked diff were unchanged. The original review CSV, datasets, experiment source, frozen splits, checkpoints and existing results were not modified. No training, inference, experiment rerun or GitHub update was performed.

**Scope limitation:** `checked` verifies CSV-to-PCAP identity, counts and timing only. The retained `malicious=1`, score 5 and reasons are Construct heuristics, not independently established malicious activity or family attribution. Most reviewed reasons are unexplained traffic outside a baseline/whitelist; sample 15 is an out-of-baseline DNS reason, and samples 7/12 include inherited paired-flow labeling. Endpoint/port/protocol alignment cannot establish whether this traffic is benign, malicious, command-and-control, or generated by the named family. These 25 selected records do not validate all dataset labels or authorize bypassing a human-review gate in the original workflow.
