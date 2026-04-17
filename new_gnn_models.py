import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (Khosla et al., 2020)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    For each anchor i in the batch, treats all samples j with the same label
    as positives and all samples k with a different label as negatives.
    Forces the backbone to learn class-discriminative representations that
    cluster essential (or non-essential) genes together regardless of species.

    L = -(1/N) * sum_i { (1/|P(i)|) * sum_{j in P(i)}
            log[ exp(sim(z_i, z_j) / τ) / sum_{k≠i} exp(sim(z_i, z_k) / τ) ] }

    Args:
        temperature: softmax temperature τ (default 0.1, lower = sharper contrast)
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : [N, D] L2-normalised embeddings (use F.normalize before calling)
            labels   : [N]    integer class labels (0 or 1)
        Returns:
            scalar loss (0.0 if no valid anchor exists)
        """
        device = features.device
        N = features.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarities (features are already L2-normalised)
        sim = torch.mm(features, features.T) / self.temperature   # [N, N]

        # Positive mask: same label, different sample
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0.0)

        # Numerical stability: subtract row max before exp
        sim_max = sim.detach().max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim - sim_max)

        # Denominator: all pairs except self
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        denom = exp_sim.masked_fill(self_mask, 0.0).sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Log-probability for each pair
        log_prob = (sim - sim_max) - torch.log(denom)   # [N, N]

        # Per-anchor loss: mean over positives
        n_pos  = pos_mask.sum(dim=1)          # [N]
        valid  = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(pos_mask * log_prob).sum(dim=1)   # [N]
        loss = loss[valid] / n_pos[valid]
        return loss.mean()


# ---------------------------------------------------------------------------
# Gradient Reversal Layer — core of domain-adversarial training
# ---------------------------------------------------------------------------

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    """Multiply gradients by -alpha during backward pass (gradient reversal)."""
    return _GradReverse.apply(x, alpha)


# ---------------------------------------------------------------------------
# CrossSpeciesGNN — the main model for cross-species gene essentiality
# ---------------------------------------------------------------------------

class CrossSpeciesGNN(nn.Module):
    """
    GIN-based model for cross-species gene essentiality prediction.

    Key improvements over WeightedGCN_BioFeatures:
      1. GIN layers (strictly more expressive than GCN) with edge weights.
      2. 85-dim gene features: codon usage + dinucleotides + GC/AT skew + CpG ratio.
      3. Per-layer LayerNorm + residual connections.
      4. Dual pooling (mean + max).
      5. Species-adversarial head with gradient reversal for domain adaptation:
         the backbone is penalised if it encodes species identity, forcing it to
         learn universally-transferable essentiality features.
    """
    BIO_IN_FEATURES = 8    # [gc, %A, %T, %C, %G, purine, keto, freq]
    GENE_FEAT_DIM   = 85   # 64 tri + gc + loglen + 16 di + gc_skew + at_skew + cpg

    def __init__(
        self,
        in_features: int = 8,
        gene_feat_dim: int = 85,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        pool: str = "mean+max",
        num_species: int = 0,   # 0 = no adversarial head; >0 = number of training species
        proj_dim: int = 0,      # 0 = no contrastive head; >0 = contrastive projection dim
    ):
        super().__init__()
        self.dropout   = dropout
        self.use_dual  = (pool == "mean+max")
        self.use_gfeat = (gene_feat_dim > 0)
        self.num_species = num_species

        # Input projection: float node features → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GIN layers with edge-weight support (WeightedGINConv defined below this class)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(WeightedGINConv(gin_mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        pool_dim = hidden_dim * 2 if self.use_dual else hidden_dim

        # Gene-level feature branch
        gene_proj_dim = 0
        if self.use_gfeat:
            gene_proj_dim = hidden_dim // 2
            self.gene_proj = nn.Sequential(
                nn.Linear(gene_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, gene_proj_dim),
                nn.ReLU(),
            )

        embed_dim = pool_dim + gene_proj_dim

        # Essentiality classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Species discriminator (only when num_species > 0)
        if num_species > 0:
            self.species_disc = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_species),
            )

        # Contrastive projection head (only when proj_dim > 0)
        # Projects backbone embedding to a lower-dim space for SupCon loss.
        # A separate head keeps the classification head unaffected.
        self.proj_head = None
        if proj_dim > 0:
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim),
            )

    def _embed(self, data):
        x = self.input_proj(data.x)

        edge_index  = data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1)

        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index, edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + residual

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        if self.use_dual:
            g = torch.cat([global_mean_pool(x, batch),
                           global_max_pool(x, batch)], dim=-1)
        else:
            g = global_mean_pool(x, batch)

        if self.use_gfeat:
            gf = getattr(data, "gene_feat", None)
            if gf is not None:
                g = torch.cat([g, self.gene_proj(gf)], dim=-1)

        return g

    def forward(self, data):
        return self.classifier(self._embed(data))

    def forward_adv(self, data, alpha=1.0):
        """Return (class_logits, species_logits) for adversarial training."""
        g = self._embed(data)
        return self.classifier(g), self.species_disc(grad_reverse(g, alpha))

    def forward_all(self, data, alpha: float = 1.0):
        """Return (class_logits, species_logits_or_None, proj_emb_or_None).

        Used when combining adversarial + contrastive losses in a single pass.
        species_logits is None when no species discriminator is present.
        proj_emb is L2-normalised and None when no projection head is present.
        """
        g = self._embed(data)
        logits = self.classifier(g)

        species_logits = None
        if self.num_species > 0 and hasattr(self, 'species_disc'):
            species_logits = self.species_disc(grad_reverse(g, alpha))

        proj_emb = None
        if self.proj_head is not None:
            proj_emb = F.normalize(self.proj_head(g), dim=-1)

        return logits, species_logits, proj_emb


class WeightedGCN_BioFeatures(nn.Module):
    """
    GCN that accepts continuous biological k-mer features (no embedding table).
    This model is cross-species transferable: node features are GC content,
    nucleotide fractions, purine/keto fractions, and per-gene k-mer frequency.

    Additionally accepts gene-level codon-usage features (trinucleotide
    frequencies + GC + length) which are the strongest known cross-species
    essentiality signal.  These are projected and concatenated with the
    pooled GNN embedding before the classifier MLP.

    Architecture:
      Input projection → N x GCN layers (residual) → Mean+Max pooling
      + Gene-feature projection → concat → MLP classifier
    """
    BIO_IN_FEATURES = 8    # [gc, %A, %T, %C, %G, purine, keto, freq]
    GENE_FEAT_DIM   = 66   # 64 trinucleotide freqs + GC + log_len

    def __init__(
        self,
        in_features: int = 8,
        gene_feat_dim: int = 66,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        pool: str = "mean+max",
    ):
        super().__init__()
        self.dropout = dropout
        self.use_dual_pool = (pool == "mean+max")
        self.use_gene_feat = (gene_feat_dim > 0)

        # Replaces the Embedding table — projects float node features into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        pool_out_dim = hidden_dim * 2 if self.use_dual_pool else hidden_dim

        # Gene-level feature branch (codon usage bias)
        gene_proj_dim = 0
        if self.use_gene_feat:
            gene_proj_dim = hidden_dim // 2
            self.gene_proj = nn.Sequential(
                nn.Linear(gene_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, gene_proj_dim),
                nn.ReLU(),
            )

        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim + gene_proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, data):
        x = self.input_proj(data.x)          # float node features → hidden_dim

        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1)

        for i, conv in enumerate(self.convs):
            residual = x
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + residual                  # residual connection

        batch = getattr(
            data, "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        if self.use_dual_pool:
            g = torch.cat([global_mean_pool(x, batch),
                           global_max_pool(x, batch)], dim=-1)
        else:
            g = global_mean_pool(x, batch)

        # Append gene-level codon-usage features if available
        if self.use_gene_feat:
            gene_feat = getattr(data, "gene_feat", None)
            if gene_feat is not None:
                gf = self.gene_proj(gene_feat)
                g = torch.cat([g, gf], dim=-1)

        return self.mlp(g)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean"
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # ----- GraphSAGE layers -----
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        # ----- Global pooling -----
        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)   # binary classification
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index

        # ---- GraphSAGE layers (NO edge weights allowed) ----
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)

        # ---- Classifier ----
        out = self.mlp(g)
        return out

class WeightedGCN(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            emb_dim=128, 
            hidden_dim=128, 
            num_layers=2, 
            dropout=0.2,
            pool="mean"
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.embedding_dropout = nn.Dropout(dropout)

        # ----- GCN layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
            
        # ----- Global pooling -----
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError("pool must be 'mean', 'max', or 'sum'.")
        
        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        x = self.embedding_dropout(x)

        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1)

        # ---- GCN layers        
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            #if i < len(self.convs) - 1:
            x = F.dropout(x, self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)
        
        # ---- Classifier ----        
        out = self.mlp(g)
        return out
    
class DefaultGINModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        num_layers=2, 
        dropout=0.2,
        pool="mean"
    ):
        super(DefaultGINModel, self).__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        # ----- GIN layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            in_dim = hidden_dim

        # ----- Global pooling -----
        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]
        
        # ----- Final MLP -----        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index

        # ---- GIN layers  
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)

        # ---- Classifier ----  
        out = self.mlp(g)
        return out

class WeightedGINConv(MessagePassing):
    def __init__(self, mlp, eps=0.0, train_eps=False):
        super().__init__(aggr="add")
        self.mlp = mlp

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_weight is not None:
            edge_weight = torch.cat(
                [edge_weight, torch.ones(x.size(0), device=x.device)]
            )

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = (1 + self.eps) * x + out
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

class WeightedGIN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean",
        train_eps=True
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        # ----- Weighted GIN layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(input_dim))
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(
                WeightedGINConv(mlp, train_eps=train_eps)
            )
            input_dim = hidden_dim

        # ----- Global pooling -----
        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x = self.embedding(data.x.view(-1))
        edge_index = data.edge_index

        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        g = self.pool(x, batch)
        return self.mlp(g)

class EdgeAttrGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, dropout=0.2):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(out_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)
            edge_attr = torch.cat(
                [edge_attr, torch.ones(x.size(0), device=x.device)]
            )

        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr
        )

    def message(self, x_i, x_j, edge_attr, index):
        alpha = (x_i * self.att_dst).sum(dim=-1) + \
                (x_j * self.att_src).sum(dim=-1)

        alpha = self.leaky_relu(alpha)

        if edge_attr is not None:
            alpha = alpha * edge_attr

        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, 1)

class CustomGAT(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean"
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            self.convs.append(
                EdgeAttrGATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    dropout=dropout
                )
            )
            in_dim = hidden_dim

        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x = self.embedding(data.x.view(-1))
        edge_index = data.edge_index

        edge_attr = getattr(data, "edge_attr", None)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        g = self.pool(x, batch)
        return self.mlp(g)


class AttentiveWeightedGINConv(MessagePassing):
    def __init__(self, mlp, eps=0.0, train_eps=False, dropout=0.2):
        super().__init__(aggr="add")

        self.mlp = mlp
        self.dropout = dropout

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))

        # Single-head attention (SAFE)
        hidden_dim = mlp[0].in_features
        self.att = nn.Linear(2 * hidden_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_weight is not None:
            edge_weight = torch.cat(
                [edge_weight, torch.ones(x.size(0), device=x.device)]
            )

        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_weight=edge_weight
        )

        out = (1 + self.eps) * x + out
        return self.mlp(out)
    
    def message(self, x_i, x_j, edge_weight, index):
        # Compute attention
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = self.att(alpha)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        return alpha * x_j

class AttentiveWeightedGIN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean",
        train_eps=True
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(input_dim))
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(
                AttentiveWeightedGINConv(
                    mlp,
                    train_eps=train_eps,
                    dropout=dropout
                )
            )
            input_dim = hidden_dim

        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x = self.embedding(data.x.view(-1))
        edge_index = data.edge_index

        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        g = self.pool(x, batch)
        return self.mlp(g)

