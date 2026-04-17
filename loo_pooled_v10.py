"""
loo_pooled_v10.py — LOO with constant GRL + AUC stopping

Root cause fixed in this version:
  In v7/v8 the GRL schedule produced grl_alpha≈0.000 at epoch 1 and only
  0.222 at epoch 10.  LOO-val experiments (v9) showed best checkpoints at
  epoch 1–11 for most species — i.e. the model specialised on training-species
  patterns before the adversarial constraint ever became meaningful.

  v10 uses constant grl_alpha=1.0 so the adversarial signal is present at
  full strength from the very first gradient step.

Changes vs v9:
  - grl_alpha = 1.0  (constant, no warmup schedule)  ← primary fix
  - early_stop_metric = "auc"  (threshold-free, more stable on small val sets)
  - val set = 20% of held-out species (stratified, same as v9 lv20)
  - patience = 20  (more room now that GRL prevents early degradation)
  - Three lambda_adv variants: 0.5, 1.0, 2.0

Usage:
  python loo_pooled_v10.py --held_out arabidopsis --lambda_adv 1.0
  bash submit_all_v10.sh    # all 24 jobs in parallel
"""

import os
import argparse
import csv
import random
import torch
import matplotlib
matplotlib.use("Agg")

from collections import defaultdict
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v10 import train, test

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA       = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC        = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT        = os.path.join(SRC, "resultsMCC")
MODELS_DIR = os.path.join(SRC, "loo_models_v10")
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
ES_PATIENCE = 20         # more room — GRL prevents fast degradation
VAL_FRAC    = 0.2        # fixed at best v9 value
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
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--held_out",   required=True, choices=list(ALL_SPECIES.keys()))
parser.add_argument("--lambda_adv", type=float, default=1.0,
                    help="GRL adversarial weight (try 0.5, 1.0, 2.0)")
args       = parser.parse_args()
HELD_OUT   = args.held_out
LAMBDA_ADV = args.lambda_adv

# Tag: lambda 0.5→a05, 1.0→a10, 2.0→a20
adv_tag    = f"a{int(round(LAMBDA_ADV * 10)):02d}"
VERSION    = f"v10_{adv_tag}"

print("=" * 60)
print(f"LOO fold         : hold out '{HELD_OUT}'")
print(f"Version          : {VERSION}")
print(f"GRL alpha        : 1.0 (constant — no schedule)")
print(f"lambda_adv       : {LAMBDA_ADV}")
print(f"Early stop on    : AUC (threshold-free)")
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

held_graphs = build_dataset_bio(fasta_held, labels_held, k=K)
normalize_gene_feat_inplace(held_graphs)

val_graphs, test_graphs = stratified_split(held_graphs, VAL_FRAC)
n_val_ess  = sum(int(g.y) for g in val_graphs)
n_test_ess = sum(int(g.y) for g in test_graphs)
print(f"  Val  set (20%)  : {len(val_graphs)} genes, {n_val_ess} essential")
print(f"  Test set (80%)  : {len(test_graphs)} genes, {n_test_ess} essential")


# ---------------------------------------------------------------------------
# Build training pool
# ---------------------------------------------------------------------------
train_graphs        = []
species_id          = 0
train_species_names = []

for name, (fasta, labels) in ALL_SPECIES.items():
    if name == HELD_OUT:
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
    print(f"  {name} (id={species_id}): {len(graphs)} graphs, {sum(raw_labels)} essentials → TRAIN")
    train_graphs.extend(graphs)
    train_species_names.append(name)
    species_id += 1

num_train_species = species_id
print(f"\nTotal training graphs : {len(train_graphs)}")
print(f"Training species      : {num_train_species} {train_species_names}")


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
    num_species               = num_train_species,
    lambda_adv                = LAMBDA_ADV,
    lambda_con                = 0.0,
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
    train_prior               = TRAIN_PRIOR,
    early_stop_metric         = "auc",
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

print(f"\n{VERSION} — trained on all except '{HELD_OUT}':")
print(res["metrics"].to_string(index=False))


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
metrics  = res["metrics"].iloc[0].to_dict()
csv_path = os.path.join(OUT, f"loo_{VERSION}_{HELD_OUT}_results.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "held_out_species", "train_species", "n_train_species",
        "version", "lambda_adv", "val_frac", "n_val",
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
        "lambda_adv":       LAMBDA_ADV,
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
