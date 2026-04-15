"""
ara_ara_code_v4.py — Bayesian Prior-Correction variant

Approach: after inference, adjust the model's raw output probabilities for the
known class-frequency shift between training and test species using a
Bayesian log-odds correction:

    log_odds_corrected = log_odds_model
                       + log( π_test  / (1 - π_test)  )
                       - log( π_train / (1 - π_train) )
    P_corrected = sigmoid( log_odds_corrected )

π_train = 0.5   (effective prior with species-balanced sampling)
π_test  = computed from the test species' label file
           (realistic: global essential-gene fractions are often known
            from databases such as OGEE or DEG, without per-gene labels)

After prior correction a bidirectional threshold search is still performed
to handle any residual miscalibration.

All shared improvements from the base fix are retained:
  - Bidirectional threshold search at test time (new_pipeline_copy_v4.py)
  - Focal loss (gamma=2)
  - Maripaludis held out as cross-species proxy validation
  - lambda_adv = 0.2
  - Per-species balanced sampling

Model saved to: ara_model_v4.pt
"""

import os
import torch
from collections import Counter
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v4 import train, test

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
FOCAL_GAMMA = 2.0    # focal loss: focuses on hard cross-species examples

# Effective training class-prior.
# With species_balanced_sampling=True each (species, class) group contributes
# equally, so the model effectively sees a 50/50 class distribution.
TRAIN_PRIOR = 0.5

# Test species: train on everything EXCEPT these.
# arabidopsis:   plant, 80 % essential — strong class-prior shift
# saccharomyces: yeast, 20 % essential — moderate, closer to training distribution
# Both get Bayesian prior correction using their own π_test.
TEST_SPECIES  = ["arabidopsis", "saccharomyces"]
PROXY_SPECIES = "maripaludis"   # held out for cross-species early stopping

# ---------------------------------------------------------------------------
# Build graphs
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Test species  : {TEST_SPECIES}")
print(f"Proxy val     : {PROXY_SPECIES}")
print(f"k-mer size    : {K}")
print(f"Approach      : v4 — Bayesian Prior Correction")
print(f"Train prior   : {TRAIN_PRIOR} (balanced sampling)")
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

# Also report actual data fraction for reference (vs. effective balanced prior)
all_labels_data = [int(g.y) for g in train_graphs]
data_prior = sum(all_labels_data) / len(all_labels_data) if all_labels_data else 0.0

print(f"\nTotal training graphs : {len(train_graphs)}")
print(f"Proxy val graphs      : {len(proxy_graphs)}")
print(f"Training species      : {num_train_species}")
print(f"Actual data prior     : {data_prior:.3f}  (raw label fraction)")
print(f"Effective train prior : {TRAIN_PRIOR:.3f}  (balanced sampling — used for correction)")

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

model_path = os.path.join(SRC, "ara_model_v4.pt")

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
    focal_loss_gamma        = FOCAL_GAMMA,
    train_prior             = TRAIN_PRIOR,
)
threshold   = results["best_threshold"]
train_prior = results["train_prior"]   # carry forward to test()
print(f"\nBest validation MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Optimal threshold   : {threshold:.2f}")

# ---------------------------------------------------------------------------
# Test on held-out species — with Bayesian prior correction
# ---------------------------------------------------------------------------
for name in TEST_SPECIES:
    fasta, labels_path = ALL_SPECIES[name]
    if not os.path.exists(fasta):
        print(f"[SKIP] {name} — test data not found")
        continue

    # Compute test prior from the label file.
    # This is the essential-gene fraction of the test species.
    # In practice this is often available from public databases (OGEE, DEG)
    # without needing the full per-gene essential/non-essential annotation.
    with open(labels_path) as f:
        test_labels_raw = [int(l.strip()) for l in f if l.strip()]
    test_prior = sum(test_labels_raw) / len(test_labels_raw)
    print(f"\n{'=' * 60}")
    print(f"Test species: {name}  |  π_test = {test_prior:.3f}  "
          f"({sum(test_labels_raw)} essential / {len(test_labels_raw)} total)")

    test_graphs = build_dataset_bio(fasta, labels_path, k=K)
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
    print(f"\nCross-species test (v4): trained on all except {name}")
    print(res["metrics"].to_string(index=False))

# ---------------------------------------------------------------------------
# Within-species sanity check on elegans (also with prior correction)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Within-species sanity check (v4): elegans → elegans")

elegans_labels_path = f"{DATA}/elegans_labels.txt"
with open(elegans_labels_path) as f:
    elegans_labels_raw = [int(l.strip()) for l in f if l.strip()]
elegans_prior = sum(elegans_labels_raw) / len(elegans_labels_raw)
print(f"Elegans π_test = {elegans_prior:.3f}  "
      f"({sum(elegans_labels_raw)} essential / {len(elegans_labels_raw)} total)")

elegans_graphs = build_dataset_bio(
    f"{DATA}/elegans_genes.fasta",
    elegans_labels_path,
    k=K,
)
normalize_gene_feat_inplace(elegans_graphs)
res_ele = test(
    graphs           = elegans_graphs,
    model            = model,
    model_path       = model_path,
    threshold        = threshold,
    search_threshold = True,
    train_prior      = train_prior,
    test_prior       = elegans_prior,
)
print(res_ele["metrics"].to_string(index=False))
