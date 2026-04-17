"""
loo_pooled_v9.py — LOO cross-species evaluation with LOO-val early stopping

Priority 1 fix: the validation set used for early stopping comes from the
held-out species itself (stratified val_frac%), NOT from a random 80/20
split of the training species.

Root cause addressed:
  In v7/v8 the proxy-val set (sampled from training species) produced
  val MCCs of 0.27–0.71, while test MCC on the held-out species was only
  0.06–0.19 (a 3–12× gap). Early stopping on proxy-val selects a checkpoint
  optimal for training-species distributions. Using a small fraction of the
  held-out species for val aligns the stopping signal with the actual target.

Experiment design:
  Training set : all 7 other species (unchanged)
  Val set      : val_frac% of held-out species (stratified by label)
  Test set     : remaining (1 - val_frac)% of held-out species

  Hyperparameters: v7 settings (best mean LOO test MCC in prior experiments)
    - lambda_adv = 0.2, dropout = 0.3, no SupCon loss

Usage:
  python loo_pooled_v9.py --held_out arabidopsis --val_frac 0.1
  python loo_pooled_v9.py --held_out arabidopsis --val_frac 0.2

  bash submit_all_v9.sh   # launch all 16 jobs in parallel
"""

import os
import argparse
import csv
import random
import torch
import matplotlib
matplotlib.use("Agg")  # headless on SLURM

from collections import defaultdict
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v4 import train, test

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA       = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC        = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT        = os.path.join(SRC, "resultsMCC")
MODELS_DIR = os.path.join(SRC, "loo_models_v9")
os.makedirs(OUT,        exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters (v7 settings — best mean LOO test MCC)
# ---------------------------------------------------------------------------
K           = 4
HIDDEN_DIM  = 256
NUM_LAYERS  = 4
DROPOUT     = 0.3
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 3e-4
ES_PATIENCE = 30
LAMBDA_ADV  = 0.2
LAMBDA_CON  = 0.0   # SupCon disabled — didn't improve over v7
FOCAL_GAMMA = 2.0
TRAIN_PRIOR = 0.5

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
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--held_out", required=True, choices=list(ALL_SPECIES.keys()))
parser.add_argument("--val_frac", type=float, default=0.1,
                    help="Fraction of held-out species used for val (default 0.1)")
args    = parser.parse_args()
HELD_OUT = args.held_out
VAL_FRAC = args.val_frac

# Derive short tag for filenames: 0.1 → "lv10", 0.2 → "lv20", etc.
VERSION_TAG = f"lv{int(round(VAL_FRAC * 100)):02d}"

print("=" * 60)
print(f"LOO fold         : hold out '{HELD_OUT}'")
print(f"Val strategy     : LOO-val — {int(VAL_FRAC*100)}% of held-out species (stratified)")
print(f"Test set         : remaining {int((1-VAL_FRAC)*100)}% of held-out species")
print(f"k-mer size       : {K}")
print(f"Setting          : v9 ({VERSION_TAG}) — GRL + Focal + Bayesian prior correction")
print(f"  lambda_adv     : {LAMBDA_ADV}")
print(f"  dropout        : {DROPOUT}")
print(f"  epochs         : {EPOCHS}, patience={ES_PATIENCE}")
print(f"Device           : {device}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Stratified split — preserves essential/non-essential ratio in val and test
# ---------------------------------------------------------------------------
def stratified_split(graphs, val_frac, seed=42):
    """
    Split graphs into (val, test) with stratification by label.
    Ensures at least 1 sample per class in val.
    """
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
# Build held-out species dataset and split into val + test
# ---------------------------------------------------------------------------
fasta_held, labels_held = ALL_SPECIES[HELD_OUT]

with open(labels_held) as f:
    held_labels_raw = [int(l.strip()) for l in f if l.strip()]
held_prior = sum(held_labels_raw) / len(held_labels_raw)
n_held_ess = sum(held_labels_raw)

print(f"\n########## Building held-out species graphs: {HELD_OUT}")
print(f"  Total genes      : {len(held_labels_raw)}")
print(f"  Essential genes  : {n_held_ess} ({held_prior:.3f})")

held_graphs = build_dataset_bio(fasta_held, labels_held, k=K)
normalize_gene_feat_inplace(held_graphs)

val_graphs, test_graphs = stratified_split(held_graphs, VAL_FRAC)

n_val_ess  = sum(int(g.y) for g in val_graphs)
n_test_ess = sum(int(g.y) for g in test_graphs)
print(f"  Val  set ({int(VAL_FRAC*100):2d}%)   : {len(val_graphs)} genes, {n_val_ess} essential ({n_val_ess/len(val_graphs):.3f})")
print(f"  Test set ({int((1-VAL_FRAC)*100):2d}%)   : {len(test_graphs)} genes, {n_test_ess} essential ({n_test_ess/len(test_graphs):.3f})")


# ---------------------------------------------------------------------------
# Build training pool (all species except held_out)
# ---------------------------------------------------------------------------
train_graphs        = []
species_id          = 0
train_species_names = []

for name, (fasta, labels) in ALL_SPECIES.items():
    if name == HELD_OUT:
        continue
    if not os.path.exists(fasta) or not os.path.exists(labels):
        print(f"  [SKIP] {name} — data files not found")
        continue
    with open(labels) as f:
        raw_labels = [int(l.strip()) for l in f if l.strip()]
    n_pos = sum(raw_labels)
    if n_pos == 0:
        print(f"  [SKIP] {name} — 0 positive labels")
        continue
    graphs = build_dataset_bio(fasta, labels, k=K, species_id=species_id)
    normalize_gene_feat_inplace(graphs)
    print(f"  {name} (id={species_id}): {len(graphs)} graphs, {n_pos} essentials → TRAIN")
    train_graphs.extend(graphs)
    train_species_names.append(name)
    species_id += 1

num_train_species = species_id
all_labels        = [int(g.y) for g in train_graphs]
data_prior        = sum(all_labels) / len(all_labels)

print(f"\nTotal training graphs : {len(train_graphs)}")
print(f"Training species      : {num_train_species} {train_species_names}")
print(f"Actual data prior     : {data_prior:.3f}")
print(f"Effective train prior : {TRAIN_PRIOR:.3f}  (balanced sampling)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = CrossSpeciesGNN(
    in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
    gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    pool          = "mean+max",
    num_species   = num_train_species,
    proj_dim      = 128,
).to(device)

model_path = os.path.join(MODELS_DIR, f"loo_model_{HELD_OUT}_{VERSION_TAG}.pt")


# ---------------------------------------------------------------------------
# Train — val_graphs comes from held-out species (LOO-val strategy)
# ---------------------------------------------------------------------------
print(f"\n*** VAL SET: {len(val_graphs)} genes from '{HELD_OUT}' ({VERSION_TAG}) ***")
print(f"*** Early stopping signal is now aligned with the test distribution ***\n")

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
    val_graphs                = val_graphs,   # KEY CHANGE: held-out species val
    val_split                 = 0.2,          # only used if val_graphs is None
    num_species               = num_train_species,
    lambda_adv                = LAMBDA_ADV,
    lambda_con                = LAMBDA_CON,
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
    train_prior               = TRAIN_PRIOR,
)

threshold   = results["best_threshold"]
train_prior = results["train_prior"]

print(f"\nBest Val MCC  : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Threshold     : {threshold:.2f}")
print(f"(Val set was {len(val_graphs)} genes from held-out '{HELD_OUT}')")


# ---------------------------------------------------------------------------
# Test on held-out test portion (1 - val_frac)
# ---------------------------------------------------------------------------
test_prior = n_test_ess / len(test_graphs)

print(f"\n{'='*60}")
print(f"Testing on held-out test set: {HELD_OUT}")
print(f"  n_test={len(test_graphs)}, n_essential={n_test_ess}, π_test={test_prior:.3f}")

res = test(
    graphs           = test_graphs,
    model            = model,
    model_path       = model_path,
    threshold        = threshold,
    search_threshold = True,
    train_prior      = train_prior,
    test_prior       = test_prior,
)

print(f"\nLOO test ({VERSION_TAG}) — trained on all except '{HELD_OUT}':")
print(res["metrics"].to_string(index=False))


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
metrics  = res["metrics"].iloc[0].to_dict()
csv_path = os.path.join(OUT, f"loo_v9_{VERSION_TAG}_{HELD_OUT}_results.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "held_out_species", "train_species", "n_train_species",
        "val_strategy", "val_frac", "n_val", "n_val_essential",
        "best_val_mcc", "best_epoch",
        "n_test", "MCC", "AUC", "Sensitivity", "Specificity",
        "Precision", "Recall", "Accuracy", "pi_test", "Threshold",
        "TN", "FP", "FN", "TP",
    ])
    writer.writeheader()
    writer.writerow({
        "held_out_species": HELD_OUT,
        "train_species":    "|".join(train_species_names),
        "n_train_species":  num_train_species,
        "val_strategy":     f"loo_val_{VERSION_TAG}",
        "val_frac":         VAL_FRAC,
        "n_val":            len(val_graphs),
        "n_val_essential":  n_val_ess,
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
print(f"LOO fold '{HELD_OUT}' ({VERSION_TAG}) complete.")
