import math
import torch
from torch_geometric.data import Data
from collections import Counter

# Fixed, sorted ordering for all 64 trinucleotides and 16 dinucleotides — species-agnostic
_BASES = 'ACGT'
_ALL_TRIS = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
_ALL_DIS  = [a + b     for a in _BASES for b in _BASES]  # 16 dinucleotides


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------

def read_fasta(filepath):
    records = []
    with open(filepath, 'r') as f:
        header, seq_lines = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq_lines).upper()))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, ''.join(seq_lines).upper()))
    return records


def read_labels(filepath):
    with open(filepath, 'r') as f:
        return [int(l.strip()) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Vocabulary-based dataset (original pipeline, kept intact)
# ---------------------------------------------------------------------------

def build_vocab(seqs, k, min_count=1):
    cnt = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if set(kmer) <= set('ATCGN'):
                cnt[kmer] += 1
    items = sorted([kmer for kmer, c in cnt.items() if c >= min_count])
    vocab = {"<UNK>": 0}
    for i, kmer in enumerate(items, start=1):
        vocab[kmer] = i
    return vocab


def seq_to_graph(seq, k, vocab, bidirectional=False, normalize=True):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    if len(kmers) == 0:
        return None
    unk = vocab.get('<UNK>', 0)

    unique_kmers = list(dict.fromkeys(kmers))
    node_index_map = {kmer: idx for idx, kmer in enumerate(unique_kmers)}

    node_ids = [vocab.get(kmer, unk) for kmer in unique_kmers]
    x = torch.tensor(node_ids, dtype=torch.long).unsqueeze(1)

    edge_counter = Counter()
    for i in range(len(kmers) - 1):
        src = node_index_map[kmers[i]]
        dst = node_index_map[kmers[i + 1]]
        edge_counter[(src, dst)] += 1
        if bidirectional:
            edge_counter[(dst, src)] += 1

    edges, weights = zip(*edge_counter.items())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)

    if normalize:
        row, col = edge_index
        out_degree = torch.zeros(len(unique_kmers), dtype=torch.float)
        for (src, _), w in zip(edges, weights):
            out_degree[src] += w

        norm_weights = []
        for (src, _), w in zip(edges, weights):
            if out_degree[src] > 0:
                norm_weights.append(w / out_degree[src])
            else:
                norm_weights.append(0.0)
        edge_attr = torch.tensor(norm_weights, dtype=torch.float)

    edge_attr = edge_attr.unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.kmers = unique_kmers
    return data


def build_dataset(fasta_path, labels_path, k, vocab=None):
    print("########## Start building garphs from seqs...")

    records = read_fasta(fasta_path)
    labels = read_labels(labels_path)

    seqs = [r[1] for r in records]
    gene_ids = [r[0] for r in records]

    if vocab is None:
        vocab = build_vocab(seqs, k)

    graphs = []
    for seq, lab, gid in zip(seqs, labels, gene_ids):
        g = seq_to_graph(seq, k, vocab)
        if g is None:
            continue

        g.y = torch.tensor([lab], dtype=torch.long)
        g.gene_id = gid
        graphs.append(g)

    print("########## Graph Built!")
    return graphs, vocab


# ---------------------------------------------------------------------------
# Bio-feature graph building (species-agnostic, no vocabulary needed)
# ---------------------------------------------------------------------------

def kmer_bio_features(kmer):
    """
    7 species-agnostic node features for a single k-mer:
      [gc_content, frac_A, frac_T, frac_C, frac_G, purine_frac, keto_frac]

    - GC / nucleotide fractions: basic composition, universal across species.
    - Purine fraction (A+G): known to differ between essential/non-essential genes.
    - Keto fraction (G+T): orthogonal compositional axis.
    Unknown bases (N) are distributed evenly across A/T/C/G.
    """
    k = len(kmer)
    cnt = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
    n_unknown = 0
    for base in kmer:
        if base in cnt:
            cnt[base] += 1
        else:
            n_unknown += 1
    spread = n_unknown / 4.0
    fa = (cnt['A'] + spread) / k
    ft = (cnt['T'] + spread) / k
    fc = (cnt['C'] + spread) / k
    fg = (cnt['G'] + spread) / k
    gc     = fc + fg
    purine = fa + fg          # A + G
    keto   = fg + ft          # G + T
    return [gc, fa, ft, fc, fg, purine, keto]


def compute_gene_features(seq):
    """
    85 gene-level features that are species-agnostic:
      - 64 normalized trinucleotide (3-mer) frequencies   (codon usage bias)
      -  1 whole-gene GC content
      -  1 log-normalized gene length
      - 16 dinucleotide frequencies                        (CpG, neighbour bias)
      -  1 GC skew  = (G−C)/(G+C)
      -  1 AT skew  = (A−T)/(A+T)
      -  1 CpG ratio (observed/expected, capped at 5 then normalised to [0,1])

    Dinucleotide frequencies and skew measures capture strand-composition
    asymmetries that correlate with essentiality across very distant species
    (plants, animals, bacteria) without being species-specific.
    """
    n = max(len(seq), 1)

    # Basic base counts
    cA = seq.count('A') / n
    cT = seq.count('T') / n
    cC = seq.count('C') / n
    cG = seq.count('G') / n
    gc = cC + cG
    log_len = math.log(len(seq) + 1) / 12.0   # log(~150k)/12 ≈ 1, roughly [0,1]

    # Trinucleotide frequencies (64)
    tri_counts = Counter()
    for i in range(len(seq) - 2):
        tri = seq[i:i+3]
        if set(tri) <= set('ATCG'):
            tri_counts[tri] += 1
    total_tri = sum(tri_counts.values())
    tri_freq = [tri_counts.get(t, 0) / max(total_tri, 1) for t in _ALL_TRIS]

    # Dinucleotide frequencies (16)
    di_counts = Counter()
    for i in range(len(seq) - 1):
        di = seq[i:i+2]
        if set(di) <= set('ATCG'):
            di_counts[di] += 1
    total_di = sum(di_counts.values())
    di_freq = [di_counts.get(d, 0) / max(total_di, 1) for d in _ALL_DIS]

    # GC skew and AT skew
    gc_sum = cG + cC
    at_sum = cA + cT
    gc_skew = (cG - cC) / gc_sum if gc_sum > 0 else 0.0
    at_skew = (cA - cT) / at_sum if at_sum > 0 else 0.0

    # CpG observed/expected ratio (capped at 5, normalised to [0,1])
    cpg_obs = di_counts.get('CG', 0) / max(total_di, 1)
    cpg_exp = cC * cG
    cpg_ratio = min((cpg_obs / cpg_exp) if cpg_exp > 0 else 0.0, 5.0) / 5.0

    return tri_freq + [gc, log_len] + di_freq + [gc_skew, at_skew, cpg_ratio]
    # 64 + 2 + 16 + 3 = 85 features


def seq_to_graph_bio(seq, k, bidirectional=False, normalize=True):
    """
    Build a k-mer co-occurrence graph with species-agnostic node features.

    Node features per k-mer (7 values):
      [gc_content, %A, %T, %C, %G, purine_frac, keto_frac]
    Plus normalized k-mer frequency in this gene appended → 8 features total.

    A gene_feat tensor (66 values) for global codon-usage features is
    also attached to the Data object.
    """
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)
             if set(seq[i:i+k]) <= set('ATCGN')]
    if len(kmers) < 2:
        return None

    unique_kmers = list(dict.fromkeys(kmers))
    node_index_map = {kmer: idx for idx, kmer in enumerate(unique_kmers)}

    # Normalized frequency of each unique k-mer in this gene
    kmer_counts = Counter(kmers)
    total_kmers = len(kmers)

    # 7 bio features + 1 frequency = 8 node features
    node_features = [
        kmer_bio_features(kmer) + [kmer_counts[kmer] / total_kmers]
        for kmer in unique_kmers
    ]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_counter = Counter()
    for i in range(len(kmers) - 1):
        src = node_index_map[kmers[i]]
        dst = node_index_map[kmers[i + 1]]
        edge_counter[(src, dst)] += 1
        if bidirectional:
            edge_counter[(dst, src)] += 1

    if len(edge_counter) == 0:
        return None

    edges, weights = zip(*edge_counter.items())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)

    if normalize:
        out_degree = torch.zeros(len(unique_kmers), dtype=torch.float)
        for (src, _), w in zip(edges, weights):
            out_degree[src] += w
        norm_weights = [
            w / out_degree[src] if out_degree[src] > 0 else 0.0
            for (src, _), w in zip(edges, weights)
        ]
        edge_attr = torch.tensor(norm_weights, dtype=torch.float)

    edge_attr = edge_attr.unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.kmers = unique_kmers
    return data


def build_dataset_bio(fasta_path, labels_path, k, species_id=None):
    """
    Build a graph dataset using biological k-mer features + gene-level codon features.
    No vocabulary is needed — fully safe for cross-species evaluation.

    Args:
        species_id: optional int; stored as graph.species_id for domain-adversarial training.
    """
    print(f"########## Building bio-feature graphs from {fasta_path} (k={k})...")
    records = read_fasta(fasta_path)
    labels = read_labels(labels_path)

    seqs = [r[1] for r in records]
    gene_ids = [r[0] for r in records]

    graphs = []
    for seq, lab, gid in zip(seqs, labels, gene_ids):
        g = seq_to_graph_bio(seq, k)
        if g is None:
            continue
        # Gene-level features: shape [1, 85]
        # Stored as 2D so PyG batching stacks to [batch_size, 85], not flat concat.
        g.gene_feat = torch.tensor(
            compute_gene_features(seq), dtype=torch.float
        ).unsqueeze(0)
        g.y = torch.tensor([lab], dtype=torch.long)
        g.gene_id = gid
        if species_id is not None:
            g.species_id = torch.tensor([species_id], dtype=torch.long)
        graphs.append(g)

    print(f"########## {len(graphs)} graphs built!")
    return graphs


# ---------------------------------------------------------------------------
# Per-species gene-feature normalization (z-score, in-place)
# ---------------------------------------------------------------------------

def normalize_gene_feat_inplace(graphs):
    """
    Z-score normalize the gene_feat tensor across the provided list of graphs
    (in-place).  Call once per species on training data, and once on test data,
    so that features represent *deviation from species average* rather than
    absolute values.  This is the key step that makes codon-usage and GC-based
    features transferable across phylogenetically distant species.

    Returns (mean, std) computed from these graphs, in case you need to apply
    the same normalization elsewhere.
    """
    all_feats = torch.cat([g.gene_feat for g in graphs], dim=0)  # [N, D]
    mean = all_feats.mean(dim=0, keepdim=True)                    # [1, D]
    std  = all_feats.std(dim=0, keepdim=True).clamp(min=1e-8)     # [1, D]
    for g in graphs:
        g.gene_feat = (g.gene_feat - mean) / std
    return mean, std
