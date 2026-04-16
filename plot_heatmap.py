"""
plot_heatmap.py — Build and plot the cross-species MCC / AUC heatmap

Reads:
  resultsMCC/single_{species}_results.csv    — single-species training rows (8 files)
  resultsMCC/loo_{species}_results.csv       — LOO pooled training row (8 files)

Produces:
  resultsMCC/heatmap_mcc.png    — MCC heatmap  (rows=train, cols=test)
  resultsMCC/heatmap_auc.png    — AUC heatmap
  resultsMCC/heatmap_combined.png — side-by-side MCC + AUC
  resultsMCC/matrix_mcc.csv     — MCC matrix as CSV
  resultsMCC/matrix_auc.csv     — AUC matrix as CSV

Row order:
  Single-species rows (8): elegans, melanogaster, musculus, maripaludis,
                            bacillus, sapiens, arabidopsis, saccharomyces
  Pooled LOO row (1):       "Pooled (LOO)" — each column j = trained on all except j,
                            tested on j

Usage:
  python plot_heatmap.py
  python plot_heatmap.py --metric mcc        # only MCC heatmap
  python plot_heatmap.py --metric auc        # only AUC heatmap
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[INFO] seaborn not found — using matplotlib for heatmap")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src"
OUT = os.path.join(SRC, "resultsMCC")

SPECIES_ORDER = [
    "elegans",
    "melanogaster",
    "musculus",
    "maripaludis",
    "bacillus",
    "sapiens",
    "arabidopsis",
    "saccharomyces",
]

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

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["mcc", "auc", "both"], default="both")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load single-species matrix rows
# ---------------------------------------------------------------------------
# mcc_matrix[train_idx][test_idx], auc_matrix[train_idx][test_idx]
N = len(SPECIES_ORDER)
mcc_matrix_single = np.full((N, N), np.nan)
auc_matrix_single = np.full((N, N), np.nan)

missing_single = []
for i, train_sp in enumerate(SPECIES_ORDER):
    csv_path = os.path.join(OUT, f"single_{train_sp}_results.csv")
    if not os.path.exists(csv_path):
        missing_single.append(train_sp)
        print(f"  [MISSING] {csv_path}")
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_sp = row["test_species"]
            if test_sp not in SPECIES_ORDER:
                continue
            j = SPECIES_ORDER.index(test_sp)
            mcc_matrix_single[i, j] = float(row["MCC"])
            auc_matrix_single[i, j] = float(row["AUC"])

# ---------------------------------------------------------------------------
# Load LOO results (pooled row)
# ---------------------------------------------------------------------------
mcc_loo = np.full(N, np.nan)
auc_loo = np.full(N, np.nan)

missing_loo = []
for j, sp in enumerate(SPECIES_ORDER):
    csv_path = os.path.join(OUT, f"loo_{sp}_results.csv")
    if not os.path.exists(csv_path):
        missing_loo.append(sp)
        print(f"  [MISSING] {csv_path}")
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mcc_loo[j] = float(row["MCC"])
            auc_loo[j] = float(row["AUC"])

# ---------------------------------------------------------------------------
# Report availability
# ---------------------------------------------------------------------------
n_single_available = sum(1 for sp in SPECIES_ORDER
                         if os.path.exists(os.path.join(OUT, f"single_{sp}_results.csv")))
n_loo_available    = sum(1 for sp in SPECIES_ORDER
                         if os.path.exists(os.path.join(OUT, f"loo_{sp}_results.csv")))

print(f"\nSingle-species rows available : {n_single_available} / {N}")
print(f"LOO rows available            : {n_loo_available} / {N}")

if n_single_available == 0 and n_loo_available == 0:
    print("\n[ERROR] No result files found. Run the training jobs first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Assemble combined matrix (N+1 rows: N single + 1 LOO)
# ---------------------------------------------------------------------------
combined_mcc = np.vstack([mcc_matrix_single, mcc_loo.reshape(1, N)])
combined_auc = np.vstack([auc_matrix_single, auc_loo.reshape(1, N)])

row_labels = [SPECIES_LABELS.get(sp, sp) for sp in SPECIES_ORDER] + ["Pooled (LOO)"]
col_labels = [SPECIES_LABELS.get(sp, sp) for sp in SPECIES_ORDER]

# ---------------------------------------------------------------------------
# Save matrices as CSV
# ---------------------------------------------------------------------------
def save_matrix_csv(matrix, row_labels, col_labels, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train \\ test"] + col_labels)
        for i, rl in enumerate(row_labels):
            row_vals = []
            for v in matrix[i]:
                row_vals.append(f"{v:.4f}" if not np.isnan(v) else "")
            writer.writerow([rl] + row_vals)
    print(f"Saved → {path}")

save_matrix_csv(combined_mcc, row_labels, col_labels,
                os.path.join(OUT, "matrix_mcc.csv"))
save_matrix_csv(combined_auc, row_labels, col_labels,
                os.path.join(OUT, "matrix_auc.csv"))

# ---------------------------------------------------------------------------
# Print tables to stdout
# ---------------------------------------------------------------------------
def print_matrix(matrix, row_labels, col_labels, title):
    col_w = 16
    row_w = 20
    header = f"\n{'':>{row_w}} " + " ".join(f"{c[:col_w]:>{col_w}}" for c in col_labels)
    print(f"\n{'='*len(header)}")
    print(f" {title}")
    print('='*len(header))
    print(header)
    sep = f"{'':>{row_w}} " + " ".join("-"*col_w for _ in col_labels)
    print(sep)
    for i, rl in enumerate(row_labels):
        vals = []
        for v in matrix[i]:
            vals.append(f"{v:>{col_w}.4f}" if not np.isnan(v) else f"{'—':>{col_w}}")
        line = f"{rl[:row_w]:>{row_w}} " + " ".join(vals)
        if rl == "Pooled (LOO)":
            print("-" * len(line))
        print(line)

print_matrix(combined_mcc, row_labels, col_labels, "MCC Matrix (rows=train, cols=test)")
print_matrix(combined_auc, row_labels, col_labels, "AUC Matrix (rows=train, cols=test)")

# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------
def make_heatmap(matrix, row_labels, col_labels, title, cmap, vmin, vmax,
                 out_path, figsize=(14, 10), fmt=".3f"):
    fig, ax = plt.subplots(figsize=figsize)

    # Replace NaN with a sentinel for display
    display = np.where(np.isnan(matrix), -9999, matrix)

    if HAS_SEABORN:
        # Mask NaN cells
        mask = np.isnan(matrix)
        annot = np.where(np.isnan(matrix), "",
                         np.vectorize(lambda x: f"{x:{fmt}}")(matrix))
        sns.heatmap(
            matrix,
            mask=mask,
            annot=annot,
            fmt="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "MCC" if "MCC" in title else "AUC",
                      "shrink": 0.8},
            xticklabels=col_labels,
            yticklabels=row_labels,
        )
    else:
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="MCC" if "MCC" in title else "AUC", shrink=0.8)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.3f}",
                            ha="center", va="center",
                            color="black" if matrix[i, j] > (vmin + vmax) / 2 else "white",
                            fontsize=8)

    # Draw separator line before the LOO row
    ax.axhline(y=len(row_labels) - 1, color="black", linewidth=2.5)

    # Highlight diagonal of the single-species block
    for i in range(N):
        if not np.isnan(matrix[i, i]):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="black",
                                        linewidth=2))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Test species", fontsize=11)
    ax.set_ylabel("Train species", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if args.metric in ("mcc", "both"):
    make_heatmap(
        combined_mcc, row_labels, col_labels,
        title="Cross-species gene essentiality — MCC\n(rows=train, cols=test; diagonal boxed = within-species)",
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        out_path=os.path.join(OUT, "heatmap_mcc.png"),
    )

if args.metric in ("auc", "both"):
    make_heatmap(
        combined_auc, row_labels, col_labels,
        title="Cross-species gene essentiality — AUC\n(rows=train, cols=test; diagonal boxed = within-species)",
        cmap="Blues",
        vmin=0.5, vmax=1.0,
        out_path=os.path.join(OUT, "heatmap_auc.png"),
    )

# Side-by-side combined figure
if args.metric == "both" and n_single_available + n_loo_available > 0:
    fig, axes = plt.subplots(1, 2, figsize=(26, 10))
    nrows = len(row_labels)
    ncols = len(col_labels)

    for ax, matrix, title, cmap, vmin, vmax, cbarlabel in [
        (axes[0], combined_mcc, "MCC", "YlOrRd", 0.0, 1.0, "MCC"),
        (axes[1], combined_auc, "AUC", "Blues",  0.5, 1.0, "AUC"),
    ]:
        if HAS_SEABORN:
            mask = np.isnan(matrix)
            annot = np.where(np.isnan(matrix), "",
                             np.vectorize(lambda x: f"{x:.3f}")(matrix))
            sns.heatmap(
                matrix,
                mask=mask,
                annot=annot,
                fmt="",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5,
                linecolor="white",
                ax=ax,
                cbar_kws={"label": cbarlabel, "shrink": 0.8},
                xticklabels=col_labels,
                yticklabels=row_labels,
            )
        else:
            masked = np.ma.masked_invalid(matrix)
            im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, label=cbarlabel, shrink=0.8)
            ax.set_xticks(range(ncols))
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(nrows))
            ax.set_yticklabels(row_labels, fontsize=8)

        ax.axhline(y=nrows - 1, color="black", linewidth=2.5)
        for i in range(N):
            if not np.isnan(matrix[i, i]):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="black",
                                            linewidth=2))
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

print("\nDone.")
