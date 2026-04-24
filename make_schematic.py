import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(28, 20), facecolor="#0A1628")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 28)
ax.set_ylim(0, 20)
ax.axis("off")

# ── palette ──────────────────────────────────────────────────────────────────
BG       = "#0A1628"
PANEL    = "#112240"
BORDER   = "#1E4D7B"
BLUE_GIN = "#1565C0"
BLUE_GIN2= "#42A5F5"
POOL_C   = "#1B5E20"
POOL_C2  = "#66BB6A"
GRL_C    = "#B71C1C"
GRL_C2   = "#EF5350"
CLS_C    = "#E65100"
CLS_C2   = "#FFA726"
INP_C    = "#0D2137"
V5_C     = "#2E7D32"
V7_C     = "#1565C0"
V9_C     = "#6A1B9A"
V10_C    = "#BF360C"
NEXT_C   = "#4A148C"
ARROW_C  = "#4FC3F7"
TXT      = "#E0E0E0"
TXT_DIM  = "#78909C"
TXT_HI   = "#FFD54F"
WHITE    = "#FFFFFF"
RED_BAD  = "#EF9A9A"
GRN_OK   = "#A5D6A7"


def rbox(ax, cx, cy, w, h, fc, ec=WHITE, lw=1.8, r=0.2, alpha=1.0, z=3):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)

def t(ax, x, y, s, sz=9, c=TXT, bold=False, ha="center", va="center", z=5, it=False):
    ax.text(x, y, s, fontsize=sz, color=c, fontweight="bold" if bold else "normal",
            ha=ha, va=va, zorder=z, fontstyle="italic" if it else "normal",
            multialignment="center")

def arr(ax, x1, y1, x2, y2, c=ARROW_C, lw=2.0, ms=12, z=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=ms, connectionstyle="arc3,rad=0.0"),
                zorder=z)

def curved_arr(ax, x1, y1, x2, y2, c=ARROW_C, lw=1.8, ms=11, rad=0.2, z=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=z)

def vtag(ax, cx, cy, label, fc):
    rbox(ax, cx, cy, 1.2, 0.28, fc=fc, ec=WHITE, lw=0.8, r=0.1, z=8)
    t(ax, cx, cy, label, sz=7, bold=True, z=9)

def section_panel(ax, x, y, w, h, title):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=0.35",
                       fc=PANEL, ec=BORDER, lw=1.2, alpha=0.5, zorder=1)
    ax.add_patch(p)
    t(ax, x + w/2, y + h - 0.22, title, sz=9.5, c=TXT_DIM, bold=True, z=2)


# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
t(ax, 14, 19.55, "Cross-Species Gene Essentiality GNN — Architecture & Version History",
  sz=18, bold=True, c=TXT_HI)
t(ax, 14, 19.1,
  "Three sections: ① Input processing  ②  GNN Encoder (4 × GIN layers)  ③  Dual output heads + losses",
  sz=10.5, c=TXT_DIM)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
lx, ly = 23.5, 18.95
rbox(ax, lx + 1.5, ly - 1.25, 4.5, 2.8, fc=PANEL, ec=BORDER, lw=1.2, r=0.25)
t(ax, lx + 1.5, ly - 0.12, "Version Tags", sz=9, bold=True, c=TXT_DIM)
legend_items = [
    (V5_C,  "v5/v6 — Initial architecture"),
    (V7_C,  "v7/v8 — Proxy-val evaluation"),
    (V9_C,  "v9    — LOO-val fix"),
    (V10_C, "v10  — Constant GRL + AUC stop"),
    (NEXT_C,"Next  — Priority 2 (planned)"),
]
for i, (fc, lbl) in enumerate(legend_items):
    yy = ly - 0.55 - i * 0.42
    rbox(ax, lx + 0.45, yy, 0.65, 0.25, fc=fc, ec=WHITE, lw=0.7, r=0.08, z=8)
    t(ax, lx + 0.78, yy, lbl, sz=8.5, c=TXT, ha="left", z=9)


# ═══════════════════════════════════════════════════════════════════════════
# ① INPUT SECTION  x: 0.3 → 5.5
# ═══════════════════════════════════════════════════════════════════════════
section_panel(ax, 0.3, 1.8, 5.5, 16.8, "① INPUT")
IX = 2.9   # centre x

# Gene sequence
rbox(ax, IX, 17.5, 4.8, 0.75, fc=INP_C, ec=V5_C, lw=2.2)
t(ax, IX, 17.72, "Gene DNA Sequence  (FASTA)", sz=11, bold=True, c=V5_C)
t(ax, IX, 17.45, "raw nucleotide sequence — variable length", sz=8.5, c=TXT_DIM, it=True)
vtag(ax, 0.92, 17.8, "v5/v6", V5_C)
arr(ax, IX, 17.12, IX, 16.75)

# k-mer graph builder
rbox(ax, IX, 16.35, 4.8, 0.92, fc="#0A2010", ec=V5_C, lw=2.2)
t(ax, IX, 16.62, "k-mer Graph Builder", sz=11.5, bold=True, c=POOL_C2)
t(ax, IX, 16.38, "k = 4  (tested k=4 vs k=5 → k=4 kept)", sz=8.5, c=TXT)
t(ax, IX, 16.17, "Nodes = unique 4-mers  |  Edges = co-occurrence freq.", sz=8, c=TXT_DIM, it=True)
vtag(ax, 0.92, 16.7, "v5/v6", V5_C)
arr(ax, IX, 15.89, IX, 15.5)

# Bio features
rbox(ax, IX, 15.1, 4.8, 0.92, fc="#0A2010", ec=V5_C, lw=2.2)
t(ax, IX, 15.36, "Biological Node Features", sz=11.5, bold=True, c=POOL_C2)
t(ax, IX, 15.1,  "GC content · sequence length · codon bias", sz=8.5, c=TXT)
t(ax, IX, 14.9,  "nucleotide complexity score (appended to k-mer freq.)", sz=8, c=TXT_DIM, it=True)
vtag(ax, 0.92, 15.45, "v5/v6", V5_C)
arr(ax, IX, 14.64, IX, 14.27)

# Node embedding
rbox(ax, IX, 13.9, 4.8, 0.72, fc=INP_C, ec=BORDER, lw=1.5)
t(ax, IX, 14.1,  "Node Feature Vector → Linear Embed → dim 256", sz=9.5, bold=True, c=TXT)
t(ax, IX, 13.84, "4⁴ k-mer freqs + bio features  →  FC layer  →  256-d", sz=8, c=TXT_DIM, it=True)
vtag(ax, 0.92, 14.22, "v5/v6", V5_C)
arr(ax, IX, 13.54, IX, 13.2)

# separator
ax.plot([0.5, 5.7], [13.05, 13.05], color=BORDER, lw=1, ls="--", zorder=2, alpha=0.6)

# Species domain label
rbox(ax, IX, 12.65, 4.8, 0.72, fc="#1A0808", ec=GRL_C, lw=2.0)
t(ax, IX, 12.87, "Species Domain Label", sz=10, bold=True, c=GRL_C2)
t(ax, IX, 12.6,  "one-hot: one of 8 species  (used by adversarial GRL head)", sz=8.2, c=TXT_DIM, it=True)
vtag(ax, 0.92, 13.0, "v5/v6", V5_C)

# Essentiality label
rbox(ax, IX, 11.7, 4.8, 0.72, fc="#1A1000", ec=CLS_C, lw=2.0)
t(ax, IX, 11.92, "Essentiality Label", sz=10, bold=True, c=CLS_C2)
t(ax, IX, 11.65, "binary: 1 = essential · 0 = non-essential", sz=8.2, c=TXT_DIM, it=True)
vtag(ax, 0.92, 12.05, "v5/v6", V5_C)

ax.plot([0.5, 5.7], [11.25, 11.25], color=BORDER, lw=1, ls="--", zorder=2, alpha=0.6)

# Validation strategy
rbox(ax, IX, 10.5, 4.8, 1.5, fc="#0D0820", ec=V9_C, lw=2.5)
t(ax, IX, 11.05, "Validation Strategy", sz=11, bold=True, c=V9_C)
t(ax, IX, 10.78, "v7/v8:  proxy-val (from training species)  ✗", sz=8.5, c=RED_BAD)
t(ax, IX, 10.56, "        → val MCC 3–12× higher than test MCC", sz=8.2, c=RED_BAD, it=True)
t(ax, IX, 10.35, "v9:      LOO-val 10% / 20% of held-out species  ✓", sz=8.5, c=GRN_OK)
t(ax, IX, 10.13, "v10:    same LOO-val 20%", sz=8.2, c=TXT_DIM)
vtag(ax, 0.92, 10.92, "v7/v8", V7_C)
vtag(ax, 0.92, 10.38, "v9", V9_C)

# Early stopping criterion
rbox(ax, IX, 9.0, 4.8, 1.3, fc="#0D0820", ec=V10_C, lw=2.5)
t(ax, IX, 9.5,  "Early Stopping Criterion", sz=11, bold=True, c=V10_C)
t(ax, IX, 9.26, "v7–v9:  monitor val  MCC  |  patience = 20", sz=8.5, c=TXT)
t(ax, IX, 9.05, "v10:    monitor val  AUC  |  patience = 20  ✓", sz=8.5, c=GRN_OK)
t(ax, IX, 8.82, "AUC more stable than MCC on small val sets", sz=8, c=TXT_DIM, it=True)
vtag(ax, 0.92, 9.62, "v7-v9", V7_C)
vtag(ax, 0.92, 9.05, "v10", V10_C)

# Graph batch
rbox(ax, IX, 7.85, 4.8, 0.72, fc=INP_C, ec=BORDER, lw=1.5)
t(ax, IX, 8.07, "Graph Batch", sz=10, bold=True, c=TXT)
t(ax, IX, 7.8,  "batch_size = 128  |  stratified by class balance", sz=8.2, c=TXT_DIM, it=True)
vtag(ax, 0.92, 8.2, "v5/v6", V5_C)


# ═══════════════════════════════════════════════════════════════════════════
# ② GNN ENCODER  x: 6.1 → 17.5
# ═══════════════════════════════════════════════════════════════════════════
section_panel(ax, 6.1, 1.8, 11.6, 16.8, "② GNN ENCODER  (CrossSpeciesGNN)")
EX = 11.5    # centre x encoder

# Main input arrow to encoder
arr(ax, 5.3, 8.0, 6.6, 12.2, c=ARROW_C, lw=2.5)
t(ax, 5.85, 10.4, "graph\nbatch", sz=8, c=TXT_DIM, it=True)

# ── GIN blocks (stacked) ─────────────────────────────────────────────────
gin_ys = [17.35, 15.65, 13.95, 12.25]
gin_labels = ["GIN Layer 1", "GIN Layer 2", "GIN Layer 3", "GIN Layer 4"]
for i, (gy, gl) in enumerate(zip(gin_ys, gin_labels)):
    rbox(ax, EX, gy, 8.8, 1.25, fc="#071530", ec=BLUE_GIN2, lw=2.3, r=0.22)
    t(ax, EX, gy + 0.44, gl + "  —  Graph Isomorphism Network (GIN)", sz=12, bold=True, c=BLUE_GIN2)
    t(ax, EX, gy + 0.16, "x_v  ←  MLP( x_v  +  Σ_{u ∈ N(v)}  x_u )", sz=10.5, c=TXT)
    t(ax, EX, gy - 0.12, "hidden_dim = 256  |  BatchNorm → ReLU → Dropout(0.3)  |  residual connection", sz=8.5, c=TXT_DIM, it=True)
    vtag(ax, 7.0, gy + 0.54, "v5/v6", V5_C)
    if i > 0:
        arr(ax, EX, gin_ys[i-1] - 0.63, EX, gy + 0.63)

# ── Dual Pooling
rbox(ax, EX, 10.75, 8.8, 0.98, fc="#061A0C", ec=POOL_C2, lw=2.8, r=0.22)
t(ax, EX, 11.05, "Dual Graph Pooling", sz=12.5, bold=True, c=POOL_C2)
t(ax, EX, 10.78, "h_G  =  CONCAT( MeanPool(X),  MaxPool(X) )", sz=10.5, c=TXT)
t(ax, EX, 10.52, "node features 256-d  →  graph embedding 512-d", sz=8.5, c=TXT_DIM, it=True)
vtag(ax, 7.0, 11.18, "v5/v6", V5_C)
arr(ax, EX, 11.62, EX, 11.24)

# ── Projection Head (SupCon)
rbox(ax, EX, 9.55, 8.8, 0.9, fc="#0D0A1F", ec=V9_C, lw=2.3, r=0.22)
t(ax, EX, 9.79, "Supervised Contrastive Projection Head", sz=11.5, bold=True, c=V9_C)
t(ax, EX, 9.53, "MLP: 512 → 256 → 128  |  L2-normalise  |  temperature contrastive loss", sz=9, c=TXT)
vtag(ax, 7.0, 9.93, "v5/v6", V5_C)
arr(ax, EX, 10.26, EX, 10.0)

# FORK arrow label
t(ax, EX, 9.18, "Graph embedding  h_G  (512-d)  →  split into two heads", sz=8.5, c=TXT_DIM, it=True)

arr(ax, EX, 9.1, EX - 2.4, 8.6, c=CLS_C2, lw=2.5)   # → classifier
arr(ax, EX, 9.1, EX + 2.4, 8.6, c=GRL_C2, lw=2.5)   # → domain via GRL

# ── GRL block (right fork in encoder)
grl_x = EX + 3.6
rbox(ax, grl_x, 8.05, 4.2, 0.88, fc="#1A0505", ec=GRL_C2, lw=2.8, r=0.2)
t(ax, grl_x, 8.32, "Gradient Reversal Layer  (GRL)", sz=11, bold=True, c=GRL_C2)
t(ax, grl_x, 8.05, "g(h)  =  h  in forward  |  ∂L/∂h  →  −α · ∂L/∂h  in backward", sz=8.5, c=TXT)
vtag(ax, grl_x - 1.65, 8.42, "v5/v6", V5_C)
arr(ax, grl_x, 7.61, grl_x, 7.05, c=GRL_C2)

# GRL alpha history
rbox(ax, grl_x, 6.5, 4.2, 1.4, fc="#150303", ec=GRL_C2, lw=1.8, r=0.2)
t(ax, grl_x, 7.08, "GRL α schedule — version history", sz=9.5, bold=True, c=GRL_C2)
t(ax, grl_x, 6.82, "v5–v9:  α = 2/(1 + e^{−10t}) − 1   (scheduled)", sz=8.5, c=RED_BAD)
t(ax, grl_x, 6.6,  "         α ≈ 0.000 at epoch 1  →  0.222 at epoch 10", sz=8.2, c=RED_BAD, it=True)
t(ax, grl_x, 6.38, "         adversarial signal near-zero early in training  ✗", sz=8.2, c=RED_BAD, it=True)
t(ax, grl_x, 6.15, "v10:   α = 1.0  constant from epoch 1  ✓", sz=8.5, c=GRN_OK)
vtag(ax, grl_x - 1.6, 6.87, "v5-v9", V7_C)
vtag(ax, grl_x - 1.6, 6.15, "v10", V10_C)

# lambda variants
rbox(ax, grl_x, 5.2, 4.2, 0.82, fc="#150303", ec=V10_C, lw=1.8, r=0.2)
t(ax, grl_x, 5.48, "λ  variants tested  (v10)", sz=9.5, bold=True, c=V10_C)
t(ax, grl_x, 5.21, "a05: λ=0.5   |   a10: λ=1.0   |   a20: λ=2.0", sz=9, c=TXT)
vtag(ax, grl_x - 1.6, 5.52, "v10", V10_C)
arr(ax, grl_x, 5.79, grl_x, 5.62, c=GRL_C2)


# ═══════════════════════════════════════════════════════════════════════════
# ③ OUTPUT HEADS  x: 17.9 → 27.7
# ═══════════════════════════════════════════════════════════════════════════
section_panel(ax, 17.9, 1.8, 9.8, 16.8, "③ OUTPUT HEADS + LOSSES")
CLS_X  = 20.3
DOM_X  = 25.5

# ── Essentiality Classifier
rbox(ax, CLS_X, 8.1, 4.3, 0.92, fc="#1A0C00", ec=CLS_C2, lw=2.5, r=0.2)
t(ax, CLS_X, 8.38, "Essentiality Classifier", sz=11.5, bold=True, c=CLS_C2)
t(ax, CLS_X, 8.1,  "MLP: 512 → 256 → 1  |  Sigmoid", sz=9.5, c=TXT)
vtag(ax, 18.5, 8.5, "v5/v6", V5_C)
arr(ax, CLS_X, EX - 2.4, CLS_X, 8.56, c=CLS_C2, lw=2)

# Focal loss + Bayes
rbox(ax, CLS_X, 6.8, 4.3, 1.08, fc="#1A0C00", ec=CLS_C2, lw=2.2, r=0.2)
t(ax, CLS_X, 7.18, "Focal Loss  +  Bayesian Prior Correction", sz=10.5, bold=True, c=CLS_C2)
t(ax, CLS_X, 6.92, "FL(p_t) = −(1 − p_t)^γ · log(p_t)   γ = 2", sz=9.5, c=TXT)
t(ax, CLS_X, 6.7,  "train prior π = 0.5  →  corrected at test time", sz=8.5, c=TXT_DIM, it=True)
vtag(ax, 18.5, 7.28, "v5/v6", V5_C)
arr(ax, CLS_X, 7.64, CLS_X, 7.35, c=CLS_C2)

# Threshold search
rbox(ax, CLS_X, 5.65, 4.3, 0.88, fc="#1A0C00", ec=CLS_C, lw=1.8, r=0.2)
t(ax, CLS_X, 5.89, "Decision Threshold Search", sz=10.5, bold=True, c=CLS_C2)
t(ax, CLS_X, 5.62, "grid search at test time  →  maximise MCC", sz=8.5, c=TXT_DIM, it=True)
vtag(ax, 18.5, 6.02, "v5/v6", V5_C)
arr(ax, CLS_X, 6.26, CLS_X, 6.09, c=CLS_C2)

# Output metrics
rbox(ax, CLS_X, 4.65, 4.3, 0.78, fc=INP_C, ec=BORDER, lw=1.5, r=0.18)
t(ax, CLS_X, 4.88, "Output Metrics", sz=10, bold=True, c=TXT)
t(ax, CLS_X, 4.62, "MCC  |  AUC  |  Sensitivity  |  Specificity  |  Precision  |  Acc", sz=8.5, c=TXT_DIM)
arr(ax, CLS_X, 5.21, CLS_X, 5.04, c=CLS_C2)

# ── Domain Classifier
rbox(ax, DOM_X, 8.1, 4.3, 0.92, fc="#1A0505", ec=GRL_C2, lw=2.5, r=0.2)
t(ax, DOM_X, 8.38, "Domain Classifier", sz=11.5, bold=True, c=GRL_C2)
t(ax, DOM_X, 8.1,  "MLP: 512 → 256 → 8  |  Softmax", sz=9.5, c=TXT)
vtag(ax, 23.7, 8.5, "v5/v6", V5_C)
# connect from GRL output
curved_arr(ax, grl_x + 2.1, 5.2, DOM_X, 7.64, c=GRL_C2, lw=2.0, rad=-0.15)

# Domain CE loss
rbox(ax, DOM_X, 6.85, 4.3, 0.9, fc="#1A0505", ec=GRL_C2, lw=2.2, r=0.2)
t(ax, DOM_X, 7.12, "Domain Cross-Entropy Loss", sz=10.5, bold=True, c=GRL_C2)
t(ax, DOM_X, 6.85, "L_adv  = − Σ_s  y_s · log(p_s)", sz=9.5, c=TXT)
vtag(ax, 23.7, 7.22, "v5/v6", V5_C)
arr(ax, DOM_X, 7.64, DOM_X, 7.30, c=GRL_C2)

# ── Combined loss (central)
rbox(ax, 22.8, 5.6, 9.5, 1.55, fc="#0A1020", ec=TXT_HI, lw=2.8, r=0.28)
t(ax, 22.8, 6.17, "Total Training Loss", sz=13, bold=True, c=TXT_HI)
t(ax, 22.8, 5.88, "L_total  =  L_focal  +  λ · L_adv", sz=12, c=WHITE)
t(ax, 22.8, 5.63, "optimizer: Adam  |  lr = 3×10⁻⁴  |  batch = 128  |  max 200 epochs", sz=9, c=TXT_DIM)
t(ax, 22.8, 5.42, "λ = 1.0 (v5–v9)   |   λ ∈ {0.5, 1.0, 2.0}  (v10)", sz=9, c=TXT_DIM, it=True)
vtag(ax, 18.5, 6.22, "v5/v6", V5_C)
vtag(ax, 18.5, 5.6, "v10", V10_C)
arr(ax, CLS_X, 6.26, 20.3, 5.87, c=CLS_C2, lw=1.8)
arr(ax, DOM_X, 6.40, 25.5, 5.87, c=GRL_C2, lw=1.8)

# Gradient back to encoder
curved_arr(ax, 22.8, 4.83, 11.5, 10.0, c="#546E7A", lw=1.5, rad=0.18)
t(ax, 17.8, 7.3, "back-\nprop", sz=8, c=TXT_DIM, it=True)


# ═══════════════════════════════════════════════════════════════════════════
# VERSION TIMELINE  (bottom strip)
# ═══════════════════════════════════════════════════════════════════════════
# grey strip
bg = FancyBboxPatch((0.3, 0.08), 27.4, 1.55,
                    boxstyle="round,pad=0,rounding_size=0.25",
                    fc=PANEL, ec=BORDER, lw=1.0, alpha=0.6, zorder=1)
ax.add_patch(bg)
t(ax, 14, 1.52, "EXPERIMENT TIMELINE", sz=9, c=TXT_DIM, bold=True, z=2)

# timeline horizontal arrow
ax.annotate("", xy=(27.4, 0.82), xytext=(0.6, 0.82),
            arrowprops=dict(arrowstyle="-|>", color=TXT_DIM, lw=1.5,
                            mutation_scale=14), zorder=0)

versions_tl = [
    (1.8,  V5_C,   "v5 / v6\ncfix_5–6",
     "k-mer GNN · GRL · bio features\nFirst cross-species test\nMCC ≈ 0.07–0.13"),
    (5.6,  V7_C,   "v7 / v8\nProxy-val",
     "Full LOO 8-species eval\nProxy-val stopping (train species)\nMCC = 0.119 (val gap 3–12×)"),
    (9.6,  V9_C,   "v9\nLOO-val",
     "LOO-val 10% / 20% fix  ✓\nMCC → 0.174  (+46%)\nBest epoch = 1–8  (new problem)"),
    (14.2, V10_C,  "v10\na05/a10/a20",
     "Constant GRL α=1.0\nAUC stopping\nMCC ≈ 0.175  (≈ no gain)"),
    (19.5, "#4527A0","v10 diagnosis",
     "Best epoch still =1 even λ=2.0\nGRL structurally insufficient\nEpoch-1 problem persists"),
    (24.2, NEXT_C, "Priority 2\n(next step)",
     "Phylogeny-aware training\nUse only closest relatives\nfor each held-out species"),
]

for xv, col, ver, desc in versions_tl:
    rbox(ax, xv, 1.08, 2.6, 0.44, fc=col, ec=WHITE, lw=1.0, r=0.12, z=5)
    t(ax, xv, 1.08, ver, sz=8, bold=True, c=WHITE, z=6)
    t(ax, xv, 0.5, desc, sz=7, c=TXT, z=4)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTNOTE
# ═══════════════════════════════════════════════════════════════════════════
t(ax, 27.7, 0.08,
  "LOO = Leave-One-Out  |  GRL = Gradient Reversal Layer  |  "
  "GIN = Graph Isomorphism Network  |  MCC = Matthews Correlation Coefficient",
  sz=7, c=TXT_DIM, ha="right")

plt.tight_layout(pad=0)
out = "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/newsrc/newsrc/src/GEP_Model_Schematic.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
