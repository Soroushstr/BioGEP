"""
loo_pooled_v11.py — LOO with phylogeny-aware training (Priority 2)

v10 diagnostic: constant GRL + AUC stopping gave mean MCC 0.1768 vs v9 baseline
0.1738.  Best epochs still 1–5 for 6/8 species even at lambda=2.0.  GRL alone
cannot overcome cross-species divergence; the decision rule triggers Priority 2.

Two modes:

  --phylo_mode hard
      For each held-out species, exclude training species with phylogenetic
      distance >= 3 (cross-kingdom boundary).  GRL disabled (num_species=0)
      since the remaining pool is already closely related.

  --phylo_mode soft
      Keep all 7 training species but weight their loss contribution by
      exp(-beta * dist(training_sp, held_out)).  GRL kept (num_species > 0).
      --phylo_beta controls decay (0.5 = gentle, 1.0 = moderate, 2.0 = steep).

  --phylo_mode none
      Identical to v10_a10 (all species, no weighting, GRL on, lambda_adv=1.0).
      Useful as a within-script control.

Phylogenetic distance table (kingdom-level, coarse but biology-grounded):
  1 = same domain/order  (e.g. animals vs animals)
  2 = same superkingdom  (e.g. animals vs fungi, bacteria vs archaea)
  3 = cross-superkingdom (e.g. eukaryotes vs prokaryotes, animals vs plants)

Usage:
  python loo_pooled_v11.py --held_out arabidopsis --phylo_mode hard
  python loo_pooled_v11.py --held_out elegans --phylo_mode soft --phylo_beta 1.0
  bash submit_all_v11.sh
"""

import os
import math
import argparse
import csv
import random
import torch
import matplotlib
matplotlib.use("Agg")

from collections import defaultdict
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v11 import train, test

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA       = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC        = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT        = os.path.join(SRC, "resultsMCC")
MODELS_DIR = os.path.join(SRC, "loo_models_v11")
os.makedirs(OUT,        exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
K           = 4
HIDDEN_DIM  = 256
NUM_LAYERS  = 4
DROPOUT     = 0.3
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 3e-4
ES_PATIENCE = 20
VAL_FRAC    = 0.2
FOCAL_GAMMA = 2.0
TRAIN_PRIOR = 0.5
LAMBDA_ADV  = 1.0   # for soft/none modes; irrelevant when GRL is off

ALL_SPECIES = {
    "elegans":       (f"{DATA}/elegans_genes.fasta",       f"{DATA}/elegans_labels.txt"),
    "melanogaster":  (f"{DATA}/melanogaster_genes.fasta",  f"{DATA}/melanogaster_labels.txt"),
    "musculus":      (f"{DATA}/musculus_genes.fasta",      f"{DATA}/musculus_labels.txt"),
    "maripaludis":   (f"{DATA}/maripaludis_genes.fasta",   f"{DATA}/maripaludis_labels.txt"),
    "bacillus":      (f"{DATA}/bacillus_genes.fasta",      f"{DATA}/bacillus_labels.txt"),
    "sapiens":       (f"{DATA}/sapiens_genes.fasta",       f"{DATA}/sapiens_labels.txt"),
    "arabidopsis":   (f"{DATA}/arabidopsis_genes.fasta",   f"{DATA}/arabidopsis_labels.txt"),
    "saccharomyces": (f"{DATA}/saccharomyces_genes.fasta", f"{DATA}/saccharomyces_labels.txt"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Phylogenetic distance table
# Groups:  metazoa | fungi | plantae | bacteria | archaea
# Distances: 1=same group, 2=same superkingdom, 3=cross-superkingdom
# ---------------------------------------------------------------------------
_GROUPS = {
    "elegans":       "metazoa",
    "melanogaster":  "metazoa",
    "musculus":      "metazoa",
    "sapiens":       "metazoa",
    "saccharomyces": "fungi",
    "arabidopsis":   "plantae",
    "bacillus":      "bacteria",
    "maripaludis":   "archaea",
}

_GROUP_DIST = {}
for (g1, g2), d in {
    ("metazoa",  "metazoa"):  1,
    ("metazoa",  "fungi"):    2,
    ("metazoa",  "plantae"):  3,
    ("metazoa",  "bacteria"): 3,
    ("metazoa",  "archaea"):  3,
    ("fungi",    "fungi"):    0,
    ("fungi",    "plantae"):  2,
    ("fungi",    "bacteria"): 3,
    ("fungi",    "archaea"):  3,
    ("plantae",  "plantae"):  0,
    ("plantae",  "bacteria"): 3,
    ("plantae",  "archaea"):  3,
    ("bacteria", "bacteria"): 0,
    ("bacteria", "archaea"):  2,
    ("archaea",  "archaea"):  0,
}.items():
    _GROUP_DIST[(g1, g2)] = d
    _GROUP_DIST[(g2, g1)] = d


def phylo_dist(sp1, sp2):
    if sp1 == sp2:
        return 0
    g1, g2 = _GROUPS[sp1], _GROUPS[sp2]
    return _GROUP_DIST.get((g1, g2), 3)

# Hard exclusion threshold: exclude training species with distance >= this value
HARD_THRESHOLD = 3  # exclude cross-superkingdom species

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--held_out",   required=True, choices=list(ALL_SPECIES.keys()))
parser.add_argument("--phylo_mode", choices=["hard", "soft", "none"], default="hard",
                    help="hard=exclude distant species (no GRL); "
                         "soft=weight loss by proximity (GRL on); "
                         "none=all species unweighted (GRL on, same as v10)")
parser.add_argument("--phylo_beta", type=float, default=1.0,
                    help="Decay rate for soft mode: w=exp(-beta*dist). Try 0.5, 1.0, 2.0")
args       = parser.parse_args()
HELD_OUT   = args.held_out
PHYLO_MODE = args.phylo_mode
PHYLO_BETA = args.phylo_beta

# Build version tag
if PHYLO_MODE == "hard":
    VERSION = "v11_hard"
elif PHYLO_MODE == "soft":
    beta_tag = f"b{int(round(PHYLO_BETA * 10)):02d}"
    VERSION  = f"v11_soft_{beta_tag}"
else:
    VERSION = "v11_none"

print("=" * 60)
print(f"LOO fold         : hold out '{HELD_OUT}'")
print(f"Version          : {VERSION}")
print(f"Phylo mode       : {PHYLO_MODE}")
if PHYLO_MODE == "soft":
    print(f"Phylo beta       : {PHYLO_BETA} (w = exp(-{PHYLO_BETA}*dist))")
print(f"Early stop on    : AUC")
print(f"Val set          : {int(VAL_FRAC*100)}% of held-out species (stratified)")
print(f"patience         : {ES_PATIENCE}")
print(f"Device           : {device}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(graphs, val_frac, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for i, g in enumerate(graphs):
        by_label[int(g.y)].append(i)
    val_idx = set()
    for label, indices in by_label.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac))
        val_idx.update(shuffled[:n_val])
    val_graphs  = [graphs[i] for i in range(len(graphs)) if i in val_idx]
    test_graphs = [graphs[i] for i in range(len(graphs)) if i not in val_idx]
    return val_graphs, test_graphs


# ---------------------------------------------------------------------------
# Build held-out dataset and split
# ---------------------------------------------------------------------------
fasta_held, labels_held = ALL_SPECIES[HELD_OUT]
with open(labels_held) as f:
    held_labels_raw = [int(l.strip()) for l in f if l.strip()]
held_prior = sum(held_labels_raw) / len(held_labels_raw)
n_held_ess = sum(held_labels_raw)

print(f"\n########## Building held-out species graphs: {HELD_OUT}")
print(f"  Total genes     : {len(held_labels_raw)}")
print(f"  Essential genes : {n_held_ess} ({held_prior:.3f})")
print(f"  Phylo group     : {_GROUPS[HELD_OUT]}")

held_graphs = build_dataset_bio(fasta_held, labels_held, k=K)
normalize_gene_feat_inplace(held_graphs)

val_graphs, test_graphs = stratified_split(held_graphs, VAL_FRAC)
n_val_ess  = sum(int(g.y) for g in val_graphs)
n_test_ess = sum(int(g.y) for g in test_graphs)
print(f"  Val  set (20%)  : {len(val_graphs)} genes, {n_val_ess} essential")
print(f"  Test set (80%)  : {len(test_graphs)} genes, {n_test_ess} essential")


# ---------------------------------------------------------------------------
# Build training pool — filtered or weighted based on phylo_mode
# ---------------------------------------------------------------------------
train_graphs        = []
species_id          = 0
train_species_names = []
species_id_to_name  = {}

print(f"\n########## Building training pool (phylo_mode={PHYLO_MODE})")

for name, (fasta, labels) in ALL_SPECIES.items():
    if name == HELD_OUT:
        continue

    dist = phylo_dist(name, HELD_OUT)

    # Hard mode: skip species that are cross-superkingdom distant
    if PHYLO_MODE == "hard" and dist >= HARD_THRESHOLD:
        print(f"  [PHYLO-SKIP] {name} (dist={dist} >= {HARD_THRESHOLD})")
        continue

    if not os.path.exists(fasta) or not os.path.exists(labels):
        print(f"  [SKIP] {name} — files not found")
        continue
    with open(labels) as f:
        raw_labels = [int(l.strip()) for l in f if l.strip()]
    if sum(raw_labels) == 0:
        print(f"  [SKIP] {name} — 0 positive labels")
        continue

    graphs = build_dataset_bio(fasta, labels, k=K, species_id=species_id)
    normalize_gene_feat_inplace(graphs)

    weight_str = ""
    if PHYLO_MODE == "soft":
        w = math.exp(-PHYLO_BETA * dist)
        weight_str = f"  loss_weight={w:.3f}"

    print(f"  {name} (id={species_id}, dist={dist}): {len(graphs)} graphs, "
          f"{sum(raw_labels)} essentials → TRAIN{weight_str}")

    train_graphs.extend(graphs)
    train_species_names.append(name)
    species_id_to_name[species_id] = name
    species_id += 1

num_train_species = species_id
print(f"\nTotal training graphs : {len(train_graphs)}")
print(f"Training species      : {num_train_species} {train_species_names}")

if num_train_species == 0:
    raise RuntimeError(
        f"No training species remaining for held_out={HELD_OUT} "
        f"with phylo_mode={PHYLO_MODE}. Relax HARD_THRESHOLD."
    )

# ---------------------------------------------------------------------------
# Build species_weights dict for soft mode
# ---------------------------------------------------------------------------
species_weights = None
if PHYLO_MODE == "soft":
    species_weights = {
        sp_id: math.exp(-PHYLO_BETA * phylo_dist(sp_name, HELD_OUT))
        for sp_id, sp_name in species_id_to_name.items()
    }
    print(f"\nSpecies loss weights (beta={PHYLO_BETA}):")
    for sp_id, sp_name in species_id_to_name.items():
        print(f"  {sp_name} (id={sp_id}): {species_weights[sp_id]:.4f}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Hard mode: GRL disabled (num_species=0) — pool is already closely related
# Soft/none: GRL enabled
use_grl = (PHYLO_MODE != "hard")
grl_num_species = num_train_species if use_grl else 0

model = CrossSpeciesGNN(
    in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
    gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    pool          = "mean+max",
    num_species   = grl_num_species,
    proj_dim      = 128,
).to(device)

print(f"\nGRL: {'enabled (num_species=' + str(grl_num_species) + ')' if use_grl else 'disabled'}")

model_path = os.path.join(MODELS_DIR, f"loo_model_{HELD_OUT}_{VERSION}.pt")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
results = train(
    train_graphs,
    model,
    model_path                = model_path,
    epoch_n                   = EPOCHS,
    batch_size                = BATCH_SIZE,
    learning_rate             = LR,
    early_stopping_patience   = ES_PATIENCE,
    weighted_sampling         = False,
    species_balanced_sampling = True,
    val_graphs                = val_graphs,
    val_split                 = 0.2,
    num_species               = grl_num_species,
    lambda_adv                = LAMBDA_ADV,
    lambda_con                = 0.0,
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
    train_prior               = TRAIN_PRIOR,
    early_stop_metric         = "auc",
    species_weights           = species_weights,
)

threshold   = results["best_threshold"]
train_prior = results["train_prior"]

print(f"\nBest epoch    : {results['best_epoch']}")
print(f"Best val AUC  : {results['best_val_auc']:.4f}")
print(f"Best val MCC  : {results['best_val_mcc']:.4f}")
print(f"Threshold     : {threshold:.2f}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
test_prior = n_test_ess / len(test_graphs)

print(f"\n{'='*60}")
print(f"Testing on held-out test set: {HELD_OUT}")
print(f"  n_test={len(test_graphs)}, n_essential={n_test_ess}, π={test_prior:.3f}")

res = test(
    graphs           = test_graphs,
    model            = model,
    model_path       = model_path,
    threshold        = threshold,
    search_threshold = True,
    train_prior      = train_prior,
    test_prior       = test_prior,
)

print(f"\n{VERSION} — trained on {train_species_names} (held out '{HELD_OUT}'):")
print(res["metrics"].to_string(index=False))


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
metrics  = res["metrics"].iloc[0].to_dict()
csv_path = os.path.join(OUT, f"loo_{VERSION}_{HELD_OUT}_results.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "held_out_species", "train_species", "n_train_species",
        "version", "phylo_mode", "phylo_beta", "lambda_adv", "val_frac", "n_val",
        "best_val_auc", "best_val_mcc", "best_epoch",
        "n_test", "MCC", "AUC", "Sensitivity", "Specificity",
        "Precision", "Recall", "Accuracy", "pi_test", "Threshold",
        "TN", "FP", "FN", "TP",
    ])
    writer.writeheader()
    writer.writerow({
        "held_out_species": HELD_OUT,
        "train_species":    "|".join(train_species_names),
        "n_train_species":  num_train_species,
        "version":          VERSION,
        "phylo_mode":       PHYLO_MODE,
        "phylo_beta":       PHYLO_BETA if PHYLO_MODE == "soft" else "N/A",
        "lambda_adv":       LAMBDA_ADV if use_grl else 0.0,
        "val_frac":         VAL_FRAC,
        "n_val":            len(val_graphs),
        "best_val_auc":     round(results["best_val_auc"], 4),
        "best_val_mcc":     round(results["best_val_mcc"], 4),
        "best_epoch":       results["best_epoch"],
        "n_test":           metrics["#Sample"],
        "MCC":              metrics["MCC"],
        "AUC":              metrics["AUC"],
        "Sensitivity":      metrics["Sensitivity"],
        "Specificity":      metrics["Specificity"],
        "Precision":        metrics["Precision"],
        "Recall":           metrics["Recall"],
        "Accuracy":         metrics["Accuracy"],
        "pi_test":          metrics["pi_test"],
        "Threshold":        metrics["Threshold"],
        "TN":               metrics["TN"],
        "FP":               metrics["FP"],
        "FN":               metrics["FN"],
        "TP":               metrics["TP"],
    })

print(f"\nResults saved → {csv_path}")
print(f"LOO '{HELD_OUT}' ({VERSION}) complete.")
