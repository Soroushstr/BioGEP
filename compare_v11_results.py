"""
compare_v11_results.py — Compare v9_lv20 / v10_a10 / v11_hard / v11_soft_b*

Key questions:
  1. Did hard phylo exclusion improve MCC/AUC over v10?
  2. Which soft beta value works best?
  3. Did hard exclusion free the model to train longer (best_epoch >> 1)?
  4. For which species does phylo-aware training help most?

Usage:
  python compare_v11_results.py
"""

import os, csv

SRC = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT = os.path.join(SRC, "resultsMCC")

SPECIES = [
    "arabidopsis", "bacillus", "elegans", "maripaludis",
    "melanogaster", "musculus", "saccharomyces", "sapiens",
]

VARIANTS = [
    ("v9_lv20",       "loo_v9_lv20_{sp}_results.csv",           "v9 lv20"),
    ("v10_a10",       "loo_v10_a10_{sp}_results.csv",            "v10 λ=1.0"),
    ("v11_hard",      "loo_v11_hard_{sp}_results.csv",           "v11 hard"),
    ("v11_soft_b05",  "loo_v11_soft_b05_{sp}_results.csv",       "v11 β=0.5"),
    ("v11_soft_b10",  "loo_v11_soft_b10_{sp}_results.csv",       "v11 β=1.0"),
    ("v11_soft_b20",  "loo_v11_soft_b20_{sp}_results.csv",       "v11 β=2.0"),
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
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    hdr = f"{'Species':15}"
    for _, _, label in variants:
        hdr += f" {label[:12]:>12}"
    hdr += "  Δhard vs v10  Δb10 vs v10"
    print(hdr)
    print("-" * 110)

    all_vals = {tag: [] for tag, _, _ in variants}

    for sp in species_list:
        row = f"{sp:15}"
        vals = {}
        for tag, pat, _ in variants:
            data = load(os.path.join(OUT, pat.replace("{sp}", sp)))
            v    = float(data[field]) if data else None
            vals[tag] = v
            row += f(v, 12)
            if v is not None:
                all_vals[tag].append(v)

        v10 = vals.get("v10_a10")
        for cmp_tag in ["v11_hard", "v11_soft_b10"]:
            vt = vals.get(cmp_tag)
            if vt is not None and v10 is not None:
                d = vt - v10
                row += f"  {d:+.4f}      "
            else:
                row += "  N/A          "
        print(row)

    print("-" * 110)
    means = f"{'MEAN':15}"
    for tag, _, _ in variants:
        m = sum(all_vals[tag]) / len(all_vals[tag]) if all_vals[tag] else None
        means += f(m, 12)
    print(means)


print_table("TEST MCC  (higher = better)", "MCC", VARIANTS, SPECIES)
print_table("TEST AUC  (higher = better)", "AUC", VARIANTS, SPECIES)


# Best epoch table
print(f"\n{'='*90}")
print("  BEST EPOCH  (higher → model trains longer → phylo exclusion freed learning)")
print(f"{'='*90}")
ep_variants = [
    ("v9_lv20",      "loo_v9_lv20_{sp}_results.csv",       "v9 lv20"),
    ("v10_a10",      "loo_v10_a10_{sp}_results.csv",        "v10 λ=1.0"),
    ("v11_hard",     "loo_v11_hard_{sp}_results.csv",       "v11 hard"),
    ("v11_soft_b10", "loo_v11_soft_b10_{sp}_results.csv",   "v11 β=1.0"),
]
print(f"{'Species':15}" + "".join(f" {l[:12]:>13}" for _, _, l in ep_variants))
print("-" * 70)
for sp in SPECIES:
    row = f"{sp:15}"
    for tag, pat, _ in ep_variants:
        data = load(os.path.join(OUT, pat.replace("{sp}", sp)))
        ep   = int(data["best_epoch"]) if data else None
        row += f" {ep if ep is not None else 'N/A':>13}"
    print(row)

print()
print("Expected: v11_hard epochs >> v10 (1–5) if removing distant species freed learning.")
print("If v11_hard epochs are still 1–3 even for prokaryotes (1-species training),")
print("the k-mer features lack cross-species essentiality signal (move to Priority 3).")


# Training set size table — useful for interpreting hard exclusion results
print(f"\n{'='*70}")
print("  TRAINING SPECIES (hard mode) — how many remain per held-out species")
print(f"{'='*70}")
for sp in SPECIES:
    data = load(os.path.join(OUT, f"loo_v11_hard_{sp}_results.csv"))
    if data:
        train_sps = data.get("train_species", "N/A")
        n         = data.get("n_train_species", "?")
        print(f"  {sp:<15} n={n}  [{train_sps}]")
    else:
        print(f"  {sp:<15} [not yet available]")


# Per-species phylo group annotation for interpretation
print(f"\n{'='*70}")
print("  PHYLO GROUPS (for interpreting hard exclusion)")
print(f"{'='*70}")
groups = {
    "elegans": "metazoa", "melanogaster": "metazoa",
    "musculus": "metazoa", "sapiens": "metazoa",
    "saccharomyces": "fungi", "arabidopsis": "plantae",
    "bacillus": "bacteria", "maripaludis": "archaea",
}
for sp in SPECIES:
    print(f"  {sp:<15} {groups[sp]}")
print()
print("Interpretation:")
print("  Animals (4 training species available): expect hard ≈ soft — rich training signal")
print("  arabidopsis/bacillus/maripaludis (1 training species): informative lower bound")
print("  saccharomyces (5 training species): eukaryotic pool, should be competitive")
