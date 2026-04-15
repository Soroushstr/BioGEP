"""
ara_ara_code_v8.py — Ranking Loss without Adversarial Training

Motivation: the core failure across all runs is AUC ≈ 0.5 on arabidopsis —
the model cannot rank essential genes above non-essential ones.  Two things
likely contribute to this:
  (a) The GRL removes species-specific signals that might carry cross-species
      ranking information (same hypothesis as v6).
  (b) Standard CE/focal loss optimises calibrated probabilities, not rankings.
      The model can get low loss by learning "most arabidopsis genes look like
      non-essential genes from training species" — this keeps CE low but breaks AUC.

This version directly attacks both:
  - No GRL (same as v6): preserve sequence-composition features.
  - Pairwise ranking loss (same mechanism as v2/cfix_7, but now the model
    trains for 200 epochs with proper val split): at each batch, penalise
    any (essential, non-essential) pair where the essential gene does NOT
    receive a higher predicted probability than the non-essential gene.

Why v2/cfix_7 failed despite ranking loss: it stopped at epoch 34 (proxy val).
The ranking signal needs many epochs to reshape the decision surface.  With a
proper val split and 200 epochs, the ranking loss has time to actually work.

Hyperparameters:
  - lambda_rank = 0.5  (weight of ranking loss alongside CE)
  - rank_margin = 0.1  (min required probability gap per pair)
  - max_pairs   = 2048 (increased from 1024 — more gradient signal per batch)

Test species  : arabidopsis, saccharomyces
Training      : elegans, melanogaster, musculus, maripaludis, bacillus, sapiens (6)
Val           : 20% random split from training data
Model saved   : ara_model_v8.pt
Result label  : cfix_13
"""

import os
import torch
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v2 import train, test   # v2 has PairwiseRankingLoss

DATA = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC  = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"

device = "cuda" if torch.cuda.is_available() else "cpu"

ALL_SPECIES = {
    "elegans":       (f"{DATA}/elegans_genes.fasta",       f"{DATA}/elegans_labels.txt"),
    "arabidopsis":   (f"{DATA}/arabidopsis_genes.fasta",   f"{DATA}/arabidopsis_labels.txt"),
    "saccharomyces": (f"{DATA}/saccharomyces_genes.fasta", f"{DATA}/saccharomyces_labels.txt"),
    "melanogaster":  (f"{DATA}/melanogaster_genes.fasta",  f"{DATA}/melanogaster_labels.txt"),
    "musculus":      (f"{DATA}/musculus_genes.fasta",      f"{DATA}/musculus_labels.txt"),
    "maripaludis":   (f"{DATA}/maripaludis_genes.fasta",   f"{DATA}/maripaludis_labels.txt"),
    "bacillus":      (f"{DATA}/bacillus_genes.fasta",      f"{DATA}/bacillus_labels.txt"),
    "sapiens":       (f"{DATA}/sapiens_genes.fasta",       f"{DATA}/sapiens_labels.txt"),
}

K           = 4
HIDDEN_DIM  = 256
NUM_LAYERS  = 4
DROPOUT     = 0.3
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 3e-4
ES_PATIENCE = 30
LAMBDA_RANK = 0.5     # weight of pairwise ranking loss
RANK_MARGIN = 0.1     # minimum probability gap essential > non-essential
MAX_PAIRS   = 2048    # pairs sampled per batch (doubled vs cfix_7)

TEST_SPECIES = ["arabidopsis", "saccharomyces"]

print("=" * 60)
print(f"Test species  : {TEST_SPECIES}")
print(f"Val strategy  : 80/20 random split from training data")
print(f"k-mer size    : {K}")
print(f"Approach      : v8 — Ranking Loss, No GRL")
print(f"lambda_rank   : {LAMBDA_RANK}  margin={RANK_MARGIN}  max_pairs={MAX_PAIRS}")
print("=" * 60)

train_graphs = []

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
    graphs = build_dataset_bio(fasta, labels, k=K)
    normalize_gene_feat_inplace(graphs)
    print(f"  {name}: {len(graphs)} graphs, {n_pos} essentials → TRAIN")
    train_graphs.extend(graphs)

print(f"\nTotal training graphs: {len(train_graphs)}")

model = CrossSpeciesGNN(
    in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
    gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    pool          = "mean+max",
    num_species   = 0,           # no adversarial head
).to(device)

model_path = os.path.join(SRC, "ara_model_v8.pt")

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
    val_graphs                = None,      # 80/20 random split
    val_split                 = 0.2,
    num_species               = 0,         # no adversarial
    lambda_adv                = 0.0,
    label_smoothing           = 0.0,
    lambda_rank               = LAMBDA_RANK,
    rank_margin               = RANK_MARGIN,
    rank_max_pairs            = MAX_PAIRS,
)
threshold = results["best_threshold"]
print(f"\nBest validation MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Optimal threshold   : {threshold:.2f}")

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
    print(f"Cross-species test (v8): trained on all except {name}")
    print(res["metrics"].to_string(index=False))

print(f"\n{'=' * 60}")
print("Within-species sanity check (v8): elegans → elegans")
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
