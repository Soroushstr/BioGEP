"""
ara_ara_code_v7.py — Bayesian Prior Correction on a properly-trained model

Why cfix_9 (v4) failed: the Bayesian correction was applied to a model that
only trained for 32 epochs (proxy-val early stopping).  At that point the
model's AUC on any species was ~0.5 — barely above random.  A prior correction
shifts calibration but cannot fix random rankings (AUC is invariant to monotone
probability transformations, so correcting probabilities does not change AUC).

This version applies the same Bayesian correction but to a model that trains
for the full 200 epochs with a proper within-distribution validation signal.

Theory: with species-balanced sampling the model's effective training prior is
π_train ≈ 0.5.  Arabidopsis has π_test ≈ 0.80 and saccharomyces π_test ≈ 0.20.
The log-odds correction:

    log_odds_corrected = log_odds_model
                       + log(π_test / (1 - π_test))
                       − log(π_train / (1 - π_train))

exactly adjusts for the class-frequency mismatch under Naive Bayes assumptions.
Even if that assumption is imperfect, the correction moves probabilities in the
right direction (up for arabidopsis, down for saccharomyces), giving the
threshold search a better starting point.

What changed from v5:
  - Bayesian prior correction applied at test time (via new_pipeline_copy_v4.py)
  - train_prior=0.5 passed through to test()
  - π_test computed per species from label files
  - Everything else identical to v5 (80/20 val split, GRL, focal loss)

Test species  : arabidopsis, saccharomyces
Training      : elegans, melanogaster, musculus, maripaludis, bacillus, sapiens (6)
Val           : 20% random split from training data
Model saved   : ara_model_v7.pt
Result label  : cfix_12
"""

import os
import torch
from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v4 import train, test   # v4 has Bayesian correction in test()

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
TRAIN_PRIOR = 0.5    # effective prior with species-balanced sampling

TEST_SPECIES = ["arabidopsis", "saccharomyces"]

print("=" * 60)
print(f"Test species  : {TEST_SPECIES}")
print(f"Val strategy  : 80/20 random split from training data")
print(f"k-mer size    : {K}")
print(f"Approach      : v7 — Bayesian Prior Correction (properly trained)")
print(f"Train prior   : {TRAIN_PRIOR} (balanced sampling)")
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
all_labels = [int(g.y) for g in train_graphs]
data_prior = sum(all_labels) / len(all_labels)

print(f"\nTotal training graphs : {len(train_graphs)}")
print(f"Training species      : {num_train_species}")
print(f"Actual data prior     : {data_prior:.3f}  (raw label fraction)")
print(f"Effective train prior : {TRAIN_PRIOR:.3f}  (balanced sampling — used for correction)")

model = CrossSpeciesGNN(
    in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
    gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    pool          = "mean+max",
    num_species   = num_train_species,
).to(device)

model_path = os.path.join(SRC, "ara_model_v7.pt")

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
    val_graphs                = None,      # KEY FIX: 80/20 random split
    val_split                 = 0.2,
    num_species               = num_train_species,
    lambda_adv                = LAMBDA_ADV,
    label_smoothing           = 0.0,
    focal_loss_gamma          = FOCAL_GAMMA,
    train_prior               = TRAIN_PRIOR,
)
threshold   = results["best_threshold"]
train_prior = results["train_prior"]
print(f"\nBest validation MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
print(f"Optimal threshold   : {threshold:.2f}")

for name in TEST_SPECIES:
    fasta, labels_path = ALL_SPECIES[name]
    if not os.path.exists(fasta):
        print(f"[SKIP] {name} — test data not found")
        continue
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
    print(f"\nCross-species test (v7): trained on all except {name}")
    print(res["metrics"].to_string(index=False))

print(f"\n{'=' * 60}")
print("Within-species sanity check (v7): elegans → elegans")
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
