import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import wilcoxon


parser = argparse.ArgumentParser()
parser.add_argument("results_directory", type=Path)
parser.add_argument("--output", type=Path, default=Path("paired_open_set_auroc.csv"))
args = parser.parse_args()

results = defaultdict(lambda: defaultdict(dict))

for path in args.results_directory.rglob("*.json"):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue
    if not isinstance(report, dict) or report.get("method") not in {"ce_msp", "supcon_proto"}:
        continue
    if "unknown_detection_auroc" not in report.get("zero_shot", {}):
        continue
    target = str(report["target_unknown_family"])
    method = report["method"]
    seed = int(report["seed"])
    if seed in results[target][method]:
        raise SystemExit(f"Duplicate result for {target}, {method}, seed {seed}: {path}")
    results[target][method][seed] = float(report["zero_shot"]["unknown_detection_auroc"])

families = ["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"]
missing = [family for family in families if not results[family].get("ce_msp") or not results[family].get("supcon_proto")]
if missing:
    raise SystemExit(f"Missing paired method reports for {', '.join(missing)}. Found {sum(len(v) for d in results.values() for v in d.values())} run reports.")

rows = []
for family in families:
    ce_runs = results[family]["ce_msp"]
    sup_runs = results[family]["supcon_proto"]
    if set(ce_runs) != set(sup_runs):
        raise SystemExit(f"Seed mismatch for {family}: CE {sorted(ce_runs)} versus SupCon {sorted(sup_runs)}")
    ce_score = mean(ce_runs.values())
    sup_score = mean(sup_runs.values())
    rows.append({"target_family": family, "n_seeds": len(ce_runs), "ce_msp_auroc": ce_score, "supcon_proto_auroc": sup_score, "supcon_minus_ce": sup_score - ce_score})

ce = [row["ce_msp_auroc"] for row in rows]
sup = [row["supcon_proto_auroc"] for row in rows]
differences = [row["supcon_minus_ce"] for row in rows]
statistic, p_value = wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="auto")

with args.output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(f"{row['target_family']:10s} CE={row['ce_msp_auroc']:.6f} SupCon={row['supcon_proto_auroc']:.6f} Difference={row['supcon_minus_ce']:+.6f} Seeds={row['n_seeds']}")

print(f"PAIRED_ROTATIONS={len(rows)}")
print(f"CE_MEAN_AUROC={mean(ce):.6f}")
print(f"SUPCON_MEAN_AUROC={mean(sup):.6f}")
print(f"MEAN_PAIRED_DIFFERENCE={mean(differences):+.6f}")
print(f"SUPCON_WINS={sum(value > 0 for value in differences)}")
print(f"CE_WINS={sum(value < 0 for value in differences)}")
print(f"TIES={sum(value == 0 for value in differences)}")
print(f"WILCOXON_W={statistic}")
print(f"WILCOXON_TWO_SIDED_P={p_value:.8f}")
print(f"OUTPUT_CSV={args.output.resolve()}")
