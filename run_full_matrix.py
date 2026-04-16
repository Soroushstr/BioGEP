"""
run_full_matrix.py — Full cross-species matrix in one sequential run

Runs (sequentially on a single GPU):
  Part 1: 8 × LOO pooled training (train on 7 species, test on held-out 1)
  Part 2: 8 × single-species training (train on 1 species, test on all 8)
  Part 3: build and save MCC/AUC heatmaps

Best setting (v7): GRL + Focal Loss (gamma=2) + species-balanced sampling
                   + Bayesian prior correction at test time

Outputs:
  resultsMCC/loo_{species}_results.csv        — 8 files
  resultsMCC/single_{species}_results.csv     — 8 files
  loo_models/loo_model_{species}.pt           — 8 saved LOO models
  single_species_models/single_{species}.pt   — 8 saved single-species models
  resultsMCC/matrix_mcc.csv
  resultsMCC/matrix_auc.csv
  resultsMCC/heatmap_mcc.png
  resultsMCC/heatmap_auc.png
  resultsMCC/heatmap_combined.png
"""

import os
import sys
import csv
import gc
import torch
import numpy as np

from seq_encoder import build_dataset_bio, normalize_gene_feat_inplace
from new_gnn_models import CrossSpeciesGNN
from new_pipeline_copy_v4 import train, test

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA      = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data"
SRC       = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT       = os.path.join(SRC, "resultsMCC")
LOO_DIR   = os.path.join(SRC, "loo_models")
SING_DIR  = os.path.join(SRC, "single_species_models")

os.makedirs(OUT,      exist_ok=True)
os.makedirs(LOO_DIR,  exist_ok=True)
os.makedirs(SING_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared hyperparameters
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
FOCAL_GAMMA = 2.0
TRAIN_PRIOR = 0.5   # effective prior with balanced sampling

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

SPECIES_ORDER = list(ALL_SPECIES.keys())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_graphs(name, fasta, labels_path, species_id=None):
    """Load graphs for one species; species_id=None means don't set it (test mode)."""
    with open(labels_path) as f:
        raw = [int(l.strip()) for l in f if l.strip()]
    n_pos = sum(raw)
    kwargs = dict(k=K)
    if species_id is not None:
        kwargs["species_id"] = species_id
    graphs = build_dataset_bio(fasta, labels_path, **kwargs)
    normalize_gene_feat_inplace(graphs)
    return graphs, raw, n_pos


# ===========================================================================
# PART 1 — LOO pooled training (8 folds)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 1 — LOO pooled training (8 folds)")
print("=" * 70)

LOO_FIELDNAMES = [
    "held_out_species", "train_species", "n_train_species",
    "best_val_mcc", "best_epoch",
    "n_test", "MCC", "AUC", "Sensitivity", "Specificity",
    "Precision", "Recall", "Accuracy", "pi_test", "Threshold",
    "TN", "FP", "FN", "TP"
]

for held_out in SPECIES_ORDER:
    print(f"\n{'='*70}")
    print(f"LOO fold: hold out '{held_out}'")
    print(f"{'='*70}")

    csv_path = os.path.join(OUT, f"loo_{held_out}_results.csv")
    if os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} already exists — delete to rerun.")
        continue

    # Build training pool (all except held_out)
    train_graphs        = []
    species_id          = 0
    train_species_names = []

    for name, (fasta, labels_path) in ALL_SPECIES.items():
        if name == held_out:
            continue
        if not os.path.exists(fasta):
            print(f"  [SKIP] {name} — data not found")
            continue
        graphs, raw, n_pos = load_graphs(name, fasta, labels_path, species_id=species_id)
        if n_pos == 0:
            print(f"  [SKIP] {name} — 0 positive labels")
            continue
        prior = n_pos / len(raw)
        print(f"  {name} (id={species_id}): {len(graphs)} graphs, "
              f"{n_pos} essentials (π={prior:.3f}) → TRAIN")
        train_graphs.extend(graphs)
        train_species_names.append(name)
        species_id += 1

    num_train_species = species_id
    all_labels  = [int(g.y) for g in train_graphs]
    data_prior  = sum(all_labels) / len(all_labels)

    print(f"\nTotal training graphs : {len(train_graphs)}")
    print(f"Training species      : {num_train_species} {train_species_names}")
    print(f"Actual data prior     : {data_prior:.3f}")
    print(f"Effective train prior : {TRAIN_PRIOR:.3f}  (balanced sampling)")

    model = CrossSpeciesGNN(
        in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
        gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
        hidden_dim    = HIDDEN_DIM,
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT,
        pool          = "mean+max",
        num_species   = num_train_species,
    ).to(device)

    model_path = os.path.join(LOO_DIR, f"loo_model_{held_out}.pt")

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
    train_prior_used = results["train_prior"]

    print(f"\nBest Val MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")
    print(f"Threshold    : {threshold:.2f}")

    # Test on held-out species
    fasta_test, labels_test_path = ALL_SPECIES[held_out]
    test_graphs, test_raw, _ = load_graphs(held_out, fasta_test, labels_test_path)
    test_prior = sum(test_raw) / len(test_raw)

    print(f"\nTesting on held-out: {held_out}  "
          f"(π={test_prior:.3f}, {sum(test_raw)}/{len(test_raw)} essential)")

    res = test(
        graphs           = test_graphs,
        model            = model,
        model_path       = model_path,
        threshold        = threshold,
        search_threshold = True,
        train_prior      = train_prior_used,
        test_prior       = test_prior,
    )
    print(res["metrics"].to_string(index=False))

    metrics = res["metrics"].iloc[0].to_dict()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOO_FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "held_out_species":  held_out,
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
    print(f"Saved → {csv_path}")

    del model, train_graphs, test_graphs
    free_gpu()


# ===========================================================================
# PART 2 — Single-species training (8 species)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 2 — Single-species training (8 species × test on all 8)")
print("=" * 70)

SINGLE_FIELDNAMES = [
    "train_species", "test_species",
    "n_train", "best_val_mcc", "best_epoch",
    "n_test", "MCC", "AUC", "Sensitivity", "Specificity",
    "Precision", "Recall", "Accuracy", "pi_train", "pi_test",
    "Threshold", "TN", "FP", "FN", "TP"
]

for train_sp in SPECIES_ORDER:
    print(f"\n{'='*70}")
    print(f"Single-species training: '{train_sp}'")
    print(f"{'='*70}")

    csv_path = os.path.join(OUT, f"single_{train_sp}_results.csv")
    if os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} already exists — delete to rerun.")
        continue

    fasta_train, labels_train_path = ALL_SPECIES[train_sp]
    if not os.path.exists(fasta_train):
        print(f"[ERROR] Training data not found: {fasta_train}")
        continue

    train_graphs, raw_labels, n_pos = load_graphs(
        train_sp, fasta_train, labels_train_path, species_id=0
    )
    if n_pos == 0:
        print(f"[ERROR] {train_sp} has 0 positive labels — skipping.")
        continue

    data_prior = n_pos / len(raw_labels)
    print(f"{train_sp}: {len(train_graphs)} graphs, {n_pos} essentials "
          f"(π={data_prior:.3f})")
    print(f"Effective train prior : {TRAIN_PRIOR:.3f}  (class-balanced sampling)")

    model = CrossSpeciesGNN(
        in_features   = CrossSpeciesGNN.BIO_IN_FEATURES,
        gene_feat_dim = CrossSpeciesGNN.GENE_FEAT_DIM,
        hidden_dim    = HIDDEN_DIM,
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT,
        pool          = "mean+max",
        num_species   = 0,    # GRL disabled for single-species
    ).to(device)

    model_path = os.path.join(SING_DIR, f"single_{train_sp}.pt")

    results = train(
        train_graphs,
        model,
        model_path                = model_path,
        epoch_n                   = EPOCHS,
        batch_size                = BATCH_SIZE,
        learning_rate             = LR,
        early_stopping_patience   = ES_PATIENCE,
        weighted_sampling         = True,
        species_balanced_sampling = False,
        val_graphs                = None,
        val_split                 = 0.2,
        num_species               = 0,
        lambda_adv                = 0.0,
        label_smoothing           = 0.0,
        focal_loss_gamma          = FOCAL_GAMMA,
        train_prior               = TRAIN_PRIOR,
    )

    threshold        = results["best_threshold"]
    train_prior_used = results["train_prior"]

    print(f"\nBest Val MCC : {results['best_val_mcc']:.4f} (epoch {results['best_epoch']})")

    rows = []
    for test_name, (fasta_t, labels_t) in ALL_SPECIES.items():
        if not os.path.exists(fasta_t):
            print(f"  [SKIP] {test_name} — test data not found")
            continue
        test_graphs, test_raw, _ = load_graphs(test_name, fasta_t, labels_t)
        test_prior = sum(test_raw) / len(test_raw)

        print(f"\n  Testing on: {test_name}  "
              f"(π={test_prior:.3f}, {sum(test_raw)}/{len(test_raw)} essential)")

        res = test(
            graphs           = test_graphs,
            model            = model,
            model_path       = model_path,
            threshold        = threshold,
            search_threshold = True,
            train_prior      = train_prior_used,
            test_prior       = test_prior,
        )
        m = res["metrics"].iloc[0].to_dict()
        marker = " ←" if test_name == train_sp else ""
        print(f"  MCC={m['MCC']:.4f}  AUC={m['AUC']:.4f}{marker}")

        rows.append({
            "train_species": train_sp,
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
        })
        del test_graphs

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SINGLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")

    del model, train_graphs
    free_gpu()


# ===========================================================================
# PART 3 — Build heatmap
# ===========================================================================
print("\n" + "=" * 70)
print("PART 3 — Building heatmap")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[INFO] seaborn not available — using matplotlib for heatmap")

SPECIES_LABELS = {
    "elegans":       "C. elegans",
    "melanogaster":  "D. melanogaster",
    "musculus":      "M. musculus",
    "maripaludis":   "M. maripaludis",
    "bacillus":      "B. subtilis",
    "sapiens":       "H. sapiens",
    "arabidopsis":   "A. thaliana",
    "saccharomyces": "S. cerevisiae",
}

N = len(SPECIES_ORDER)
mcc_single = np.full((N, N), np.nan)
auc_single = np.full((N, N), np.nan)

for i, train_sp in enumerate(SPECIES_ORDER):
    path = os.path.join(OUT, f"single_{train_sp}_results.csv")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            test_sp = row["test_species"]
            if test_sp in SPECIES_ORDER:
                j = SPECIES_ORDER.index(test_sp)
                mcc_single[i, j] = float(row["MCC"])
                auc_single[i, j] = float(row["AUC"])

mcc_loo = np.full(N, np.nan)
auc_loo = np.full(N, np.nan)

for j, sp in enumerate(SPECIES_ORDER):
    path = os.path.join(OUT, f"loo_{sp}_results.csv")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            mcc_loo[j] = float(row["MCC"])
            auc_loo[j] = float(row["AUC"])

combined_mcc = np.vstack([mcc_single, mcc_loo.reshape(1, N)])
combined_auc = np.vstack([auc_single, auc_loo.reshape(1, N)])

row_labels = [SPECIES_LABELS[sp] for sp in SPECIES_ORDER] + ["Pooled (LOO)"]
col_labels = [SPECIES_LABELS[sp] for sp in SPECIES_ORDER]

# Save CSV matrices
def save_csv(matrix, row_labels, col_labels, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train \\ test"] + col_labels)
        for i, rl in enumerate(row_labels):
            vals = [f"{v:.4f}" if not np.isnan(v) else "" for v in matrix[i]]
            w.writerow([rl] + vals)
    print(f"Saved → {path}")

save_csv(combined_mcc, row_labels, col_labels, os.path.join(OUT, "matrix_mcc.csv"))
save_csv(combined_auc, row_labels, col_labels, os.path.join(OUT, "matrix_auc.csv"))

# Print tables
def print_matrix(matrix, row_labels, col_labels, title):
    cw, rw = 16, 20
    hdr = f"\n{'':>{rw}} " + " ".join(f"{c[:cw]:>{cw}}" for c in col_labels)
    print(f"\n{'='*len(hdr)}\n {title}\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'':>{rw}} " + " ".join("-"*cw for _ in col_labels))
    for i, rl in enumerate(row_labels):
        vals = [f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'—':>{cw}}" for v in matrix[i]]
        line = f"{rl[:rw]:>{rw}} " + " ".join(vals)
        if rl == "Pooled (LOO)":
            print("-" * len(line))
        print(line)

print_matrix(combined_mcc, row_labels, col_labels, "MCC Matrix (rows=train, cols=test)")
print_matrix(combined_auc, row_labels, col_labels, "AUC Matrix (rows=train, cols=test)")

# Build heatmaps
def make_heatmap(matrix, row_labels, col_labels, title, cmap, vmin, vmax, out_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    if HAS_SEABORN:
        mask = np.isnan(matrix)
        annot = np.where(np.isnan(matrix), "",
                         np.vectorize(lambda x: f"{x:.3f}")(matrix))
        sns.heatmap(matrix, mask=mask, annot=annot, fmt="", cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                    ax=ax, cbar_kws={"shrink": 0.8},
                    xticklabels=col_labels, yticklabels=row_labels)
    else:
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                            fontsize=8,
                            color="black" if matrix[i,j] > (vmin+vmax)/2 else "white")
    ax.axhline(y=len(row_labels)-1, color="black", linewidth=2.5)
    for i in range(N):
        if not np.isnan(matrix[i, i]):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                        fill=False, edgecolor="black", linewidth=2))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Test species", fontsize=11)
    ax.set_ylabel("Train species", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

make_heatmap(combined_mcc, row_labels, col_labels,
             "Cross-species gene essentiality — MCC\n(diagonal = within-species; last row = LOO pooled)",
             "YlOrRd", 0.0, 1.0, os.path.join(OUT, "heatmap_mcc.png"))

make_heatmap(combined_auc, row_labels, col_labels,
             "Cross-species gene essentiality — AUC\n(diagonal = within-species; last row = LOO pooled)",
             "Blues", 0.5, 1.0, os.path.join(OUT, "heatmap_auc.png"))

# Combined side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(26, 10))
for ax, matrix, title, cmap, vmin, vmax in [
    (axes[0], combined_mcc, "MCC", "YlOrRd", 0.0, 1.0),
    (axes[1], combined_auc, "AUC", "Blues",  0.5, 1.0),
]:
    if HAS_SEABORN:
        mask  = np.isnan(matrix)
        annot = np.where(np.isnan(matrix), "",
                         np.vectorize(lambda x: f"{x:.3f}")(matrix))
        sns.heatmap(matrix, mask=mask, annot=annot, fmt="", cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                    ax=ax, cbar_kws={"shrink": 0.8},
                    xticklabels=col_labels, yticklabels=row_labels)
    else:
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
    ax.axhline(y=len(row_labels)-1, color="black", linewidth=2.5)
    for i in range(N):
        if not np.isnan(matrix[i, i]):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                        fill=False, edgecolor="black", linewidth=2))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Test species", fontsize=10)
    ax.set_ylabel("Train species", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

fig.suptitle("Cross-species gene essentiality prediction\n"
             "(diagonal = within-species; last row = pooled leave-one-out)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
combined_path = os.path.join(OUT, "heatmap_combined.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {combined_path}")

print("\n" + "=" * 70)
print("ALL DONE.")
print("=" * 70)
