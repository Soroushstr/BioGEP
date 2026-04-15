"""
ara_ara_code_v5.py — Fixed Baseline (proper val split + focal loss + GRL)

Root-cause fix applied here: cfix_6 through cfix_9 all early-stopped at
epoch 31-34 because maripaludis proxy-val MCC peaked at epoch 1-4 and never
recovered.  The model never trained past ~30 epochs, which is why within-species
elegans MCC crashed from 0.79 (cfix_5, 200 epochs) to 0.34 (cfix_6-9, ~32 epochs).
The proxy-val signal (a prokaryote) has no correlation with arabidopsis (a plant).

What changed from cfix_6-9:
  1. NO proxy val — val_graphs=None triggers an 80/20 random split from training
     data, giving a real training signal for early stopping.
  2. Maripaludis is moved BACK into training (6 training species instead of 5).
  3. Focal loss (gamma=2) kept — focuses on hard examples.
  4. GRL adversarial (lambda_adv=0.2) kept — domain-invariant features.
  5. Bidirectional threshold search at test time.

This is the "fixed baseline": every other change from cfix_6-9 is correct,
only the early-stopping mechanism was broken.  v6, v7, v8 build on top of this.

Test species  : arabidopsis, saccharomyces
Training      : elegans, melanogaster, musculus, maripaludis, bacillus, sapiens (6)
Val           : 20% random split from training data
Model saved   : ara_model_v5.pt
Result label  : cfix_10
"""

import os
import torch
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy import train, test

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
LAMBDA_ADV  = 0.2
FOCAL_GAMMA = 2.0

TEST_SPECIES = ["arabidopsis", "saccharomyces"]

print("=" * 60)
print(f"Test species  : {TEST_SPECIES}")
print(f"Val strategy  : 80/20 random split from training data")
print(f"k-mer size    : {K}")
print(f"Approach      : v5 — Fixed Baseline (proper val split + focal + GRL)")
print("=" * 60)

train_graphs = []
species_id   = 0

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
    graphs = build_dataset_bio(fasta, labels, k=K, species_id=species_id)
    normalize_gene_feat_inplace(graphs)
    print(f"  {name} (id={species_id}): {len(graphs)} graphs, {n_pos} essentials → TRAIN")
    train_graphs.extend(graphs)
    species_id += 1

num_train_species = species_id
print(f"\nTotal training graphs: {len(train_graphs)}")
print(f"Training species     : {num_train_species}")

model = CrossSpeciesGNN(
    in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
    gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    pool          = "mean+max",
    num_species   = num_train_species,
).to(device)

model_path = os.path.join(SRC, "ara_model_v5.pt")

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
    val_graphs                = None,      # KEY FIX: 80/20 random split, not proxy val
    val_split                 = 0.2,
    num_species               = num_train_species,
    lambda_adv                = LAMBDA_ADV,
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
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
    print(f"Cross-species test (v5): trained on all except {name}")
    print(res["metrics"].to_string(index=False))

print(f"\n{'=' * 60}")
print("Within-species sanity check (v5): elegans → elegans")
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
