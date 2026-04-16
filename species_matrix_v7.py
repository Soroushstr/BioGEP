"""
species_matrix_v7.py — Single-species training → all-species test matrix

For a given training species:
  - Train ONLY on that species (focal loss, class-balanced sampling, NO GRL)
  - Test on ALL 8 species (with Bayesian prior correction)
  - Save one row of the MCC/AUC matrix to resultsMCC/single_{train_species}_results.csv

Best setting (adapted for single-species):
  GRL is disabled (only 1 species, adversarial objective is meaningless)
  Everything else follows v7: focal gamma=2, 80/20 val split, Bayesian correction.

Usage:
  python species_matrix_v7.py --train_species elegans
  python species_matrix_v7.py --train_species arabidopsis

Run one job per training species — submit_loo_matrix.sh launches all 8 in parallel.
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

MODELS_DIR = os.path.join(SRC, "single_species_models")
os.makedirs(MODELS_DIR, exist_ok=True)

K           = 4
HIDDEN_DIM  = 256
NUM_LAYERS  = 4
DROPOUT     = 0.3
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 3e-4
ES_PATIENCE = 30
FOCAL_GAMMA = 2.0
# class-balanced weighted sampling → effective train prior = 0.5
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
parser.add_argument("--train_species", required=True,
                    choices=list(ALL_SPECIES.keys()),
                    help="Species to use for training")
args = parser.parse_args()

TRAIN_SPECIES = args.train_species

print("=" * 60)
print(f"Training species : {TRAIN_SPECIES}")
print(f"Test species     : all 8")
print(f"Val strategy     : 80/20 random split from training data")
print(f"k-mer size       : {K}")
print(f"Setting          : v7 (no GRL — single species) + Bayesian correction")
print(f"Device           : {device}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Build training dataset (single species only, species_id always = 0)
# ---------------------------------------------------------------------------
fasta_train, labels_train_path = ALL_SPECIES[TRAIN_SPECIES]

if not os.path.exists(fasta_train):
    print(f"[ERROR] Training data not found: {fasta_train}")
    sys.exit(1)

with open(labels_train_path) as f:
    raw_labels = [int(l.strip()) for l in f if l.strip()]
n_pos = sum(raw_labels)
if n_pos == 0:
    print(f"[ERROR] {TRAIN_SPECIES} has 0 positive labels — cannot train.")
    sys.exit(1)

train_graphs = build_dataset_bio(fasta_train, labels_train_path, k=K, species_id=0)
normalize_gene_feat_inplace(train_graphs)

data_prior = n_pos / len(raw_labels)
print(f"\n{TRAIN_SPECIES}: {len(train_graphs)} graphs, {n_pos} essentials")
print(f"Actual data prior     : {data_prior:.3f}")
print(f"Effective train prior : {TRAIN_PRIOR:.3f}  (class-balanced weighted sampling)")

# ---------------------------------------------------------------------------
# Model — num_species=0 disables the GRL / species discriminator
# ---------------------------------------------------------------------------
model = CrossSpeciesGNN(
    in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
    gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    pool          = "mean+max",
    num_species   = 0,       # GRL disabled — no species discriminator
).to(device)

model_path = os.path.join(MODELS_DIR, f"single_{TRAIN_SPECIES}.pt")

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
    weighted_sampling         = True,    # class-balanced (2 classes)
    species_balanced_sampling = False,   # single species — no species-level balancing
    val_graphs                = None,
    val_split                 = 0.2,
    num_species               = 0,
    lambda_adv                = 0.0,     # no adversarial loss
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
    train_prior               = TRAIN_PRIOR,
)

threshold   = results["best_threshold"]
train_prior = results["train_prior"]

print(f"\nBest Val MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Threshold    : {threshold:.2f}")

# ---------------------------------------------------------------------------
# Test on ALL 8 species (with Bayesian prior correction)
# ---------------------------------------------------------------------------
csv_path = os.path.join(OUT, f"single_{TRAIN_SPECIES}_results.csv")

fieldnames = [
    "train_species", "test_species",
    "n_train", "best_val_mcc", "best_epoch",
    "n_test", "MCC", "AUC", "Sensitivity", "Specificity",
    "Precision", "Recall", "Accuracy", "pi_train", "pi_test",
    "Threshold", "TN", "FP", "FN", "TP", "flipped"
]

rows = []
for test_name, (fasta_t, labels_t) in ALL_SPECIES.items():
    if not os.path.exists(fasta_t):
        print(f"  [SKIP] {test_name} — test data not found")
        continue
    with open(labels_t) as f:
        test_labels_raw = [int(l.strip()) for l in f if l.strip()]
    test_prior = sum(test_labels_raw) / len(test_labels_raw)

    print(f"\n{'='*60}")
    print(f"Testing on: {test_name}  "
          f"(π={test_prior:.3f}, {sum(test_labels_raw)}/{len(test_labels_raw)} essential)")

    test_graphs = build_dataset_bio(fasta_t, labels_t, k=K)
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
    m = res["metrics"].iloc[0].to_dict()
    print(f"  Trained on {TRAIN_SPECIES} → tested on {test_name}:")
    print(f"  MCC={m['MCC']:.4f}  AUC={m['AUC']:.4f}")

    rows.append({
        "train_species": TRAIN_SPECIES,
        "test_species":  test_name,
        "n_train":       len(train_graphs),
        "best_val_mcc":  round(results["best_val_mcc"], 4),
        "best_epoch":    results["best_epoch"],
        "n_test":        m["#Sample"],
        "MCC":           m["MCC"],
        "AUC":           m["AUC"],
        "Sensitivity":   m["Sensitivity"],
        "Specificity":   m["Specificity"],
        "Precision":     m["Precision"],
        "Recall":        m["Recall"],
        "Accuracy":      m["Accuracy"],
        "pi_train":      round(TRAIN_PRIOR, 4),
        "pi_test":       m["pi_test"],
        "Threshold":     m["Threshold"],
        "TN":            m["TN"],
        "FP":            m["FP"],
        "FN":            m["FN"],
        "TP":            m["TP"],
        "flipped":       "",   # recorded from stdout; threshold search handles it
    })

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults saved → {csv_path}")
print(f"\nSummary for train_species={TRAIN_SPECIES}:")
print(f"{'Test species':<15} {'MCC':>8} {'AUC':>8}")
print("-" * 35)
for r in rows:
    marker = " ←" if r["test_species"] == TRAIN_SPECIES else ""
    print(f"{r['test_species']:<15} {r['MCC']:>8.4f} {r['AUC']:>8.4f}{marker}")
