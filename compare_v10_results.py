"""
compare_v10_results.py — Compare v7 / v9_lv20 / v10_a05 / v10_a10 / v10_a20

Key questions answered:
  1. Did constant GRL (v10) improve test MCC/AUC over v9?
  2. Which lambda_adv value (0.5, 1.0, 2.0) works best?
  3. Did the val AUC stop declining (i.e. does the model now train stably)?
  4. Are best epochs later than in v9 (indicating GRL prevented early specialisation)?

Usage:
  python compare_v10_results.py
"""

import os, csv

SRC = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT = os.path.join(SRC, "resultsMCC")

SPECIES = [
    "arabidopsis", "bacillus", "elegans", "maripaludis",
    "melanogaster", "musculus", "saccharomyces", "sapiens",
]

VARIANTS = [
    ("v7",       "loo_{sp}_results.csv",             "v7 proxy-val"),
    ("v9_lv20",  "loo_v9_lv20_{sp}_results.csv",     "v9 lv20"),
    ("v10_a05",  "loo_v10_a05_{sp}_results.csv",     "v10 λ=0.5"),
    ("v10_a10",  "loo_v10_a10_{sp}_results.csv",     "v10 λ=1.0"),
    ("v10_a20",  "loo_v10_a20_{sp}_results.csv",     "v10 λ=2.0"),
]


def load(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[0] if rows else None


def f(val, w=8):
    try:    return f"{float(val):{w}.4f}"
    except: return f"{'N/A':>{w}}"


def print_table(title, field, variants, species_list):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    hdr = f"{'Species':15}"
    for _, _, label in variants:
        hdr += f" {label[:10]:>10}"
    # delta columns: v10_a05/a10/a20 vs v9_lv20
    hdr += "  Δa05 vs v9  Δa10 vs v9  Δa20 vs v9"
    print(hdr)
    print("-" * 95)

    all_vals = {tag: [] for tag, _, _ in variants}

    for sp in species_list:
        row = f"{sp:15}"
        vals = {}
        for tag, pat, _ in variants:
            data = load(os.path.join(OUT, pat.replace("{sp}", sp)))
            v    = float(data[field]) if data else None
            vals[tag] = v
            row += f(v, 10)
            if v is not None:
                all_vals[tag].append(v)

        v9  = vals.get("v9_lv20")
        for atag in ["v10_a05", "v10_a10", "v10_a20"]:
            vt = vals.get(atag)
            if vt is not None and v9 is not None:
                d = vt - v9
                row += f"  {d:+.4f}    " if atag == "v10_a05" else f"{d:+.4f}    "
            else:
                row += "  N/A        "
        print(row)

    print("-" * 95)
    means = f"{'MEAN':15}"
    for tag, _, _ in variants:
        m = sum(all_vals[tag]) / len(all_vals[tag]) if all_vals[tag] else None
        means += f(m, 10)
    print(means)


print_table("TEST MCC  (higher = better)", "MCC",      VARIANTS, SPECIES)
print_table("TEST AUC  (higher = better)", "AUC",      VARIANTS, SPECIES)


# Best epoch table — key diagnostic for whether GRL fixed early stopping
print(f"\n{'='*75}")
print("  BEST EPOCH  (higher → model trains longer → GRL prevents early specialisation)")
print(f"{'='*75}")
ep_variants = [
    ("v9_lv20",  "loo_v9_lv20_{sp}_results.csv",  "v9 lv20"),
    ("v10_a05",  "loo_v10_a05_{sp}_results.csv",  "v10 λ=0.5"),
    ("v10_a10",  "loo_v10_a10_{sp}_results.csv",  "v10 λ=1.0"),
    ("v10_a20",  "loo_v10_a20_{sp}_results.csv",  "v10 λ=2.0"),
]
print(f"{'Species':15}" + "".join(f" {l[:10]:>11}" for _, _, l in ep_variants))
print("-" * 60)
for sp in SPECIES:
    row = f"{sp:15}"
    for tag, pat, _ in ep_variants:
        data = load(os.path.join(OUT, pat.replace("{sp}", sp)))
        ep   = int(data["best_epoch"]) if data else None
        row += f" {ep if ep is not None else 'N/A':>11}"
    print(row)

print()
print("Expected: v10 best epochs >> v9 best epochs (1–8) if constant GRL worked.")
print("If v10 epochs are still 1–3, GRL lambda is still too weak or other fixes needed.")


# Val AUC at best epoch — are we stopping on a good signal now?
print(f"\n{'='*75}")
print("  VAL AUC AT BEST EPOCH  (should be close to test AUC if stopping signal is good)")
print(f"{'='*75}")
print(f"{'Species':15} {'v9 valAUC':>10} {'v9 testAUC':>11}  {'v10a10 valAUC':>14} {'v10a10 testAUC':>15}")
print("-" * 70)
for sp in SPECIES:
    v9   = load(os.path.join(OUT, f"loo_v9_lv20_{sp}_results.csv"))
    v10  = load(os.path.join(OUT, f"loo_v10_a10_{sp}_results.csv"))
    v9_vauc  = f(v9["best_val_auc"]  if v9  and "best_val_auc" in v9  else None, 10)
    v9_tauc  = f(v9["AUC"]           if v9  else None, 11)
    v10_vauc = f(v10["best_val_auc"] if v10 and "best_val_auc" in v10 else None, 14)
    v10_tauc = f(v10["AUC"]          if v10 else None, 15)
    print(f"{sp:15} {v9_vauc} {v9_tauc}  {v10_vauc} {v10_tauc}")
