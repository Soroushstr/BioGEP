"""
ara_ara_code_v6.py — No Adversarial Training

Hypothesis: the gradient reversal layer (GRL) might be HURTING cross-species
transfer to arabidopsis.  The GRL forces the backbone to remove all species-
specific signal from its embeddings.  But arabidopsis is a plant — its genes
have fundamentally different sequence composition from all 6 training species.
By erasing species-identity signals, the GRL might also erase the very
compositional features (GC content patterns, codon bias, etc.) that correlate
with essentiality in a cross-species-transferable way.

What changed from v5:
  - num_species=0  → CrossSpeciesGNN built WITHOUT a species discriminator head
  - lambda_adv=0.0 → No adversarial loss term (redundant with above, but explicit)
  - Everything else is identical to v5: proper 80/20 val split, maripaludis in
    training, focal loss (gamma=2), full 200 epochs.

If v6 > v5 on arabidopsis: the GRL was hurting transfer.
If v6 ≈ v5: the GRL had no effect (species-invariance isn't the bottleneck).
If v6 < v5: the GRL was helping with within-species generalization.

Test species  : arabidopsis, saccharomyces
Training      : elegans, melanogaster, musculus, maripaludis, bacillus, sapiens (6)
Val           : 20% random split from training data
Model saved   : ara_model_v6.pt
Result label  : cfix_11
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
FOCAL_GAMMA = 2.0

TEST_SPECIES = ["arabidopsis", "saccharomyces"]

print("=" * 60)
print(f"Test species  : {TEST_SPECIES}")
print(f"Val strategy  : 80/20 random split from training data")
print(f"k-mer size    : {K}")
print(f"Approach      : v6 — No Adversarial Training (GRL removed)")
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
    # No species_id needed — adversarial head is disabled
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
    num_species   = 0,          # KEY: no species discriminator head
).to(device)

model_path = os.path.join(SRC, "ara_model_v6.pt")

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
    lambda_adv                = 0.0,       # no adversarial loss
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
    print(f"Cross-species test (v6): trained on all except {name}")
    print(res["metrics"].to_string(index=False))

print(f"\n{'=' * 60}")
print("Within-species sanity check (v6): elegans → elegans")
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
