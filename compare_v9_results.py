"""
compare_v9_results.py — Compare LOO-val (v9) vs proxy-val (v7) results

Reads all available CSVs for v7, v9_lv10, v9_lv20 and prints:
  1. Per-species MCC/AUC table with deltas vs v7
  2. Val MCC vs Test MCC gap table (key diagnostic)
  3. Summary: mean MCC across species for each variant

Usage:
  python compare_v9_results.py
"""

import os
import csv

SRC = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT = os.path.join(SRC, "resultsMCC")

SPECIES = [
    "arabidopsis", "bacillus", "elegans", "maripaludis",
    "melanogaster", "musculus", "saccharomyces", "sapiens",
]

VARIANTS = [
    ("v7",      "loo_{sp}_results.csv",         "proxy-val (v7)"),
    ("v9_lv10", "loo_v9_lv10_{sp}_results.csv", "LOO-val 10% (v9)"),
    ("v9_lv20", "loo_v9_lv20_{sp}_results.csv", "LOO-val 20% (v9)"),
]


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def fmt(val, width=7):
    try:
        return f"{float(val):>{width}.4f}"
    except (TypeError, ValueError):
        return f"{'N/A':>{width}}"


# ---------------------------------------------------------------------------
# Table 1: Test MCC and AUC per species per variant
# ---------------------------------------------------------------------------
print("=" * 90)
print("TABLE 1 — Test MCC  (higher = better)")
print("=" * 90)
header = f"{'Species':15}"
for _, _, label in VARIANTS:
    header += f" {label[:14]:>14}"
header += f"  {'Δ lv10':>8}  {'Δ lv20':>8}"
print(header)
print("-" * 90)

all_mccs = {tag: [] for tag, _, _ in VARIANTS}

for sp in SPECIES:
    row_str = f"{sp:15}"
    mccs = {}
    for tag, pattern, _ in VARIANTS:
        path = os.path.join(OUT, pattern.replace("{sp}", sp))
        data = load_csv(path)
        mcc  = float(data["MCC"]) if data else None
        mccs[tag] = mcc
        row_str += fmt(mcc, 15)
        if mcc is not None:
            all_mccs[tag].append(mcc)

    d10 = (mccs["v9_lv10"] - mccs["v7"]) if mccs.get("v9_lv10") and mccs.get("v7") else None
    d20 = (mccs["v9_lv20"] - mccs["v7"]) if mccs.get("v9_lv20") and mccs.get("v7") else None
    row_str += f"  {(f'+{d10:.4f}' if d10 and d10>=0 else f'{d10:.4f}') if d10 is not None else 'N/A':>8}"
    row_str += f"  {(f'+{d20:.4f}' if d20 and d20>=0 else f'{d20:.4f}') if d20 is not None else 'N/A':>8}"
    print(row_str)

print("-" * 90)
means_row = f"{'MEAN':15}"
for tag, _, _ in VARIANTS:
    m = sum(all_mccs[tag]) / len(all_mccs[tag]) if all_mccs[tag] else None
    means_row += fmt(m, 15)
print(means_row)


# ---------------------------------------------------------------------------
# Table 2: AUC
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("TABLE 2 — Test AUC  (higher = better)")
print("=" * 90)
print(header.replace("MCC", "AUC"))
print("-" * 90)

all_aucs = {tag: [] for tag, _, _ in VARIANTS}

for sp in SPECIES:
    row_str = f"{sp:15}"
    aucs = {}
    for tag, pattern, _ in VARIANTS:
        path = os.path.join(OUT, pattern.replace("{sp}", sp))
        data = load_csv(path)
        auc  = float(data["AUC"]) if data else None
        aucs[tag] = auc
        row_str += fmt(auc, 15)
        if auc is not None:
            all_aucs[tag].append(auc)

    d10 = (aucs["v9_lv10"] - aucs["v7"]) if aucs.get("v9_lv10") and aucs.get("v7") else None
    d20 = (aucs["v9_lv20"] - aucs["v7"]) if aucs.get("v9_lv20") and aucs.get("v7") else None
    row_str += f"  {(f'+{d10:.4f}' if d10 and d10>=0 else f'{d10:.4f}') if d10 is not None else 'N/A':>8}"
    row_str += f"  {(f'+{d20:.4f}' if d20 and d20>=0 else f'{d20:.4f}') if d20 is not None else 'N/A':>8}"
    print(row_str)

print("-" * 90)
means_row2 = f"{'MEAN':15}"
for tag, _, _ in VARIANTS:
    m = sum(all_aucs[tag]) / len(all_aucs[tag]) if all_aucs[tag] else None
    means_row2 += fmt(m, 15)
print(means_row2)


# ---------------------------------------------------------------------------
# Table 3: Val MCC vs Test MCC gap (KEY diagnostic)
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("TABLE 3 — Val MCC vs Test MCC gap  (smaller gap = better aligned stopping signal)")
print("=" * 90)
print(f"{'Species':15} {'v7 valMCC':>10} {'v7 testMCC':>11} {'v7 gap':>8}  "
      f"{'lv10 valMCC':>12} {'lv10 testMCC':>13} {'lv10 gap':>9}  "
      f"{'lv20 valMCC':>12} {'lv20 testMCC':>13} {'lv20 gap':>9}")
print("-" * 115)

for sp in SPECIES:
    row_str = f"{sp:15}"
    for tag, pattern, _ in VARIANTS:
        path = os.path.join(OUT, pattern.replace("{sp}", sp))
        data = load_csv(path)
        if data:
            val_mcc  = float(data["best_val_mcc"])
            test_mcc = float(data["MCC"])
            gap      = val_mcc - test_mcc
            row_str += f" {val_mcc:>10.4f} {test_mcc:>11.4f} {gap:>8.4f}  "
        else:
            row_str += f" {'N/A':>10} {'N/A':>11} {'N/A':>8}  "
    print(row_str)


# ---------------------------------------------------------------------------
# Table 4: Best epoch comparison (early stopping behaviour)
# ---------------------------------------------------------------------------
print()
print("=" * 75)
print("TABLE 4 — Best epoch  (higher → stopped later → more training)")
print("=" * 75)
print(f"{'Species':15} {'v7 epoch':>10} {'lv10 epoch':>12} {'lv20 epoch':>12}")
print("-" * 55)

for sp in SPECIES:
    row_str = f"{sp:15}"
    for tag, pattern, _ in VARIANTS:
        path = os.path.join(OUT, pattern.replace("{sp}", sp))
        data = load_csv(path)
        ep = int(data["best_epoch"]) if data else None
        row_str += f" {ep if ep is not None else 'N/A':>10}"
    print(row_str)

print()
print("Note: v7 musculus best_epoch was 24 (very early stop — proxy-val signal collapsed)")
print("      LOO-val should produce more stable stopping for musculus.")
