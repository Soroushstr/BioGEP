"""
loo_pooled_v7.py — Leave-One-Out cross-species evaluation (best setting: v7)

For a given held-out species:
  - Train on all remaining 7 species (pooled, 80/20 random val split)
  - Best setting: GRL + Focal Loss (gamma=2) + species-balanced sampling
                  + Bayesian prior correction at test time
  - Test on the held-out species with Bayesian correction
  - Save MCC, AUC and full metrics to resultsMCC/loo_{held_out}_results.csv

Usage:
  python loo_pooled_v7.py --held_out arabidopsis
  python loo_pooled_v7.py --held_out elegans

Run one job per species — submit_loo_matrix.sh launches all 8 in parallel.
"""

import os
import sys
import argparse
import torch
import csv
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v4 import train, test

# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------
DATA = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC  = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT  = os.path.join(SRC, "resultsMCC")
os.makedirs(OUT, exist_ok=True)

MODELS_DIR = os.path.join(SRC, "loo_models")
os.makedirs(MODELS_DIR, exist_ok=True)

K           = 4
HIDDEN_DIM  = 256
NUM_LAYERS  = 4
DROPOUT     = 0.3
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 3e-4
ES_PATIENCE = 30
LAMBDA_ADV  = 0.2
FOCAL_GAMMA = 2.0
TRAIN_PRIOR = 0.5    # effective prior with species-balanced sampling

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
parser.add_argument("--held_out", required=True,
                    choices=list(ALL_SPECIES.keys()),
                    help="Species to hold out for testing")
args = parser.parse_args()

HELD_OUT = args.held_out

print("=" * 60)
print(f"LOO fold         : hold out '{HELD_OUT}'")
print(f"Val strategy     : 80/20 random split from training data")
print(f"k-mer size       : {K}")
print(f"Setting          : v7 — GRL + Focal + Bayesian prior correction")
print(f"Device           : {device}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Build training pool (all species except held_out)
# ---------------------------------------------------------------------------
train_graphs = []
species_id   = 0
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
all_labels = [int(g.y) for g in train_graphs]
data_prior = sum(all_labels) / len(all_labels)

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
).to(device)

model_path = os.path.join(MODELS_DIR, f"loo_model_{HELD_OUT}.pt")

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
    val_graphs                = None,
    val_split                 = 0.2,
    num_species               = num_train_species,
    lambda_adv                = LAMBDA_ADV,
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
    train_prior               = TRAIN_PRIOR,
)

threshold   = results["best_threshold"]
train_prior = results["train_prior"]

print(f"\nBest Val MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Threshold    : {threshold:.2f}")

# ---------------------------------------------------------------------------
# Test on the held-out species
# ---------------------------------------------------------------------------
fasta_test, labels_test_path = ALL_SPECIES[HELD_OUT]
with open(labels_test_path) as f:
    test_labels_raw = [int(l.strip()) for l in f if l.strip()]
test_prior = sum(test_labels_raw) / len(test_labels_raw)

print(f"\n{'='*60}")
print(f"Testing on held-out: {HELD_OUT}  "
      f"(π={test_prior:.3f}, {sum(test_labels_raw)}/{len(test_labels_raw)} essential)")

test_graphs = build_dataset_bio(fasta_test, labels_test_path, k=K)
normalize_gene_feat_inplace(test_graphs)

res = test(
    graphs           = test_graphs,
    model            = model,
    model_path       = model_path,
    threshold        = threshold,
    search_threshold = True,
    train_prior      = train_prior,
    test_prior       = test_prior,
)
print(f"\nLOO test — trained on all except {HELD_OUT}:")
print(res["metrics"].to_string(index=False))

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
metrics = res["metrics"].iloc[0].to_dict()
csv_path = os.path.join(OUT, f"loo_{HELD_OUT}_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "held_out_species", "train_species", "n_train_species",
        "best_val_mcc", "best_epoch",
        "n_test", "MCC", "AUC", "Sensitivity", "Specificity",
        "Precision", "Recall", "Accuracy", "pi_test", "Threshold",
        "TN", "FP", "FN", "TP"
    ])
    writer.writeheader()
    writer.writerow({
        "held_out_species":  HELD_OUT,
        "train_species":     "|".join(train_species_names),
        "n_train_species":   num_train_species,
        "best_val_mcc":      round(results["best_val_mcc"], 4),
        "best_epoch":        results["best_epoch"],
        "n_test":            metrics["#Sample"],
        "MCC":               metrics["MCC"],
        "AUC":               metrics["AUC"],
        "Sensitivity":       metrics["Sensitivity"],
        "Specificity":       metrics["Specificity"],
        "Precision":         metrics["Precision"],
        "Recall":            metrics["Recall"],
        "Accuracy":          metrics["Accuracy"],
        "pi_test":           metrics["pi_test"],
        "Threshold":         metrics["Threshold"],
        "TN":                metrics["TN"],
        "FP":                metrics["FP"],
        "FN":                metrics["FN"],
        "TP":                metrics["TP"],
    })

print(f"\nResults saved → {csv_path}")
print(f"LOO fold '{HELD_OUT}' complete.")
