"""
ara_ara_code_v2.py — Pairwise Ranking Loss variant

Approach: replace focal loss with a pairwise hinge ranking loss that directly
penalises batches where essential genes receive lower predicted probability than
non-essential genes.  Since the core cross-species failure is AUC < 0.5
(essential genes ranked below non-essential), this loss directly attacks the
problem during training rather than post-hoc.

All shared improvements from the base fix are retained:
  - Bidirectional threshold search at test time (new_pipeline_copy_v2.py)
  - Maripaludis held out as cross-species proxy validation
  - lambda_adv = 0.2 (reduced from 0.5)
  - Per-species balanced sampling

Hyperparameters unique to v2:
  - lambda_rank = 0.5   (weight of ranking loss term)
  - rank_margin = 0.1   (minimum probability gap between essential/non-essential)

Model saved to: ara_model_v2.pt
"""

import os
import torch
from collections import Counter
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v2 import train, test

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC  = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# All available species (fasta, labels)
# ---------------------------------------------------------------------------
ALL_SPECIES = {
    "elegans":      (f"{DATA}/elegans_genes.fasta",      f"{DATA}/elegans_labels.txt"),
    "arabidopsis":  (f"{DATA}/arabidopsis_genes.fasta",  f"{DATA}/arabidopsis_labels.txt"),
    "saccharomyces":(f"{DATA}/saccharomyces_genes.fasta",f"{DATA}/saccharomyces_labels.txt"),
    "melanogaster": (f"{DATA}/melanogaster_genes.fasta", f"{DATA}/melanogaster_labels.txt"),
    "musculus":     (f"{DATA}/musculus_genes.fasta",     f"{DATA}/musculus_labels.txt"),
    "maripaludis":  (f"{DATA}/maripaludis_genes.fasta",  f"{DATA}/maripaludis_labels.txt"),
    "bacillus":     (f"{DATA}/bacillus_genes.fasta",     f"{DATA}/bacillus_labels.txt"),
    "sapiens":      (f"{DATA}/sapiens_genes.fasta",      f"{DATA}/sapiens_labels.txt"),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K           = 4
HIDDEN_DIM  = 256
NUM_LAYERS  = 4
DROPOUT     = 0.3
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 3e-4
ES_PATIENCE = 30
LAMBDA_ADV  = 0.2    # reduced — overly aggressive invariance hurts transfer
LAMBDA_RANK = 0.5    # v2-specific: weight of pairwise ranking loss
RANK_MARGIN = 0.1    # v2-specific: minimum probability gap essential/non-essential

# Test species: train on everything EXCEPT these.
# arabidopsis: plant, 80 % essential — strong class-prior shift
# saccharomyces: yeast, 20 % essential — moderate, more similar to training
TEST_SPECIES  = ["arabidopsis", "saccharomyces"]
PROXY_SPECIES = "maripaludis"   # held out for cross-species early stopping

# ---------------------------------------------------------------------------
# Build graphs
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Test species  : {TEST_SPECIES}")
print(f"Proxy val     : {PROXY_SPECIES}")
print(f"k-mer size    : {K}")
print(f"Approach      : v2 — Pairwise Ranking Loss")
print("=" * 60)

train_graphs = []
proxy_graphs = []
species_id   = 0
num_train_species = 0

for name, (fasta, labels) in ALL_SPECIES.items():
    if name in TEST_SPECIES:
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

    if name == PROXY_SPECIES:
        graphs = build_dataset_bio(fasta, labels, k=K)
        normalize_gene_feat_inplace(graphs)
        proxy_graphs.extend(graphs)
        print(f"  {name}: {len(graphs)} graphs, {n_pos} essentials → PROXY VAL")
        continue

    graphs = build_dataset_bio(fasta, labels, k=K, species_id=species_id)
    normalize_gene_feat_inplace(graphs)
    print(f"  {name} (id={species_id}): {len(graphs)} graphs, {n_pos} essentials → TRAIN")
    train_graphs.extend(graphs)
    species_id += 1

num_train_species = species_id
print(f"\nTotal training graphs : {len(train_graphs)}")
print(f"Proxy val graphs      : {len(proxy_graphs)}")
print(f"Training species      : {num_train_species}")

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

model_path = os.path.join(SRC, "ara_model_v2.pt")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
results = train(
    train_graphs,
    model,
    model_path              = model_path,
    epoch_n                 = EPOCHS,
    batch_size              = BATCH_SIZE,
    learning_rate           = LR,
    early_stopping_patience = ES_PATIENCE,
    weighted_sampling       = False,
    species_balanced_sampling = True,
    val_graphs              = proxy_graphs if proxy_graphs else None,
    num_species             = num_train_species,
    lambda_adv              = LAMBDA_ADV,
    label_smoothing         = 0.0,
    lambda_rank             = LAMBDA_RANK,
    rank_margin             = RANK_MARGIN,
)
threshold = results["best_threshold"]
print(f"\nBest validation MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Optimal threshold   : {threshold:.2f}")

# ---------------------------------------------------------------------------
# Test on held-out species
# ---------------------------------------------------------------------------
for name in TEST_SPECIES:
    fasta, labels = ALL_SPECIES[name]
    if not os.path.exists(fasta):
        print(f"[SKIP] {name} — test data not found")
        continue
    test_graphs = build_dataset_bio(fasta, labels, k=K)
    normalize_gene_feat_inplace(test_graphs)
    res = test(
        graphs           = test_graphs,
        model            = model,
        model_path       = model_path,
        threshold        = threshold,
        search_threshold = True,
    )
    print(f"\n{'=' * 60}")
    print(f"Cross-species test (v2): trained on all except {name}")
    print(res["metrics"].to_string(index=False))

# ---------------------------------------------------------------------------
# Within-species sanity check on elegans
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Within-species sanity check (v2): elegans → elegans")
elegans_graphs = build_dataset_bio(
    f"{DATA}/elegans_genes.fasta",
    f"{DATA}/elegans_labels.txt",
    k=K,
)
normalize_gene_feat_inplace(elegans_graphs)
res_ele = test(
    graphs           = elegans_graphs,
    model            = model,
    model_path       = model_path,
    threshold        = threshold,
    search_threshold = True,
)
print(res_ele["metrics"].to_string(index=False))
