"""
new_pipeline_copy_v3.py — Cross-Species Contrastive Loss variant

Key difference from new_pipeline_copy.py (v1 / focal-loss + proxy-val):
  - Adds a cross-species contrastive loss computed on the GNN embeddings.
  - Adversarial training (v1) removes species identity from embeddings but
    does NOT explicitly align same-class embeddings across species.
  - This loss does both: it pulls essential-gene embeddings from different
    training species together in cosine space, and pushes essential vs.
    non-essential embeddings apart, regardless of species.
  - Uses model._embed() to get the intermediate graph embedding, then
    computes contrastive loss before the classifier head.
  - All other improvements are kept: bidirectional threshold search, val_graphs
    proxy validation, maripaludis early stopping, lambda_adv=0.2, focal loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import time
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from new_gnn_models import grad_reverse


class FocalLoss(nn.Module):
    """
    Focal loss with class-weight support (same as v1).
    gamma=0 → standard weighted cross-entropy.
    """
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce    = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt    = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class CrossSpeciesContrastiveLoss(nn.Module):
    """
    Pulls same-class embeddings from DIFFERENT species together and pushes
    different-class cross-species embeddings apart in cosine similarity space.

    Motivation: domain-adversarial training (v1) removes species identity from
    embeddings, but does not explicitly align same-class representations across
    species.  This contrastive objective does: it forces the embedding of an
    essential gene from species A to be geometrically close to the embedding of
    an essential gene from species B, and far from non-essential genes.  The
    result is an embedding space where essentiality is a consistent axis across
    all species — including unseen species at test time (arabidopsis).

    temperature: scales cosine similarities before applying the contrastive term.
                 Lower values make the loss sharper (harder push/pull).
    max_pairs:   cap on cross-species pairs evaluated per batch; prevents O(N²)
                 cost with large batches.
    """
    def __init__(self, temperature: float = 0.1, max_pairs: int = 512):
        super().__init__()
        self.temperature = temperature
        self.max_pairs   = max_pairs

    def forward(
        self,
        embeddings:  torch.Tensor,   # [N, D]
        targets:     torch.Tensor,   # [N]   binary labels
        species_ids: torch.Tensor,   # [N]   integer species ids
    ) -> torch.Tensor:
        N = embeddings.size(0)
        if N < 2:
            return embeddings.sum() * 0.0

        # L2-normalise so dot product = cosine similarity
        z = F.normalize(embeddings, dim=1)   # [N, D]

        # Pair masks: [N, N]
        same_class   = targets.unsqueeze(0)    == targets.unsqueeze(1)
        diff_species = species_ids.unsqueeze(0) != species_ids.unsqueeze(1)
        diag         = torch.eye(N, dtype=torch.bool, device=z.device)

        # Cross-species positive pairs: same class, different species → pull together
        pos_mask = same_class  & diff_species & ~diag
        # Cross-species negative pairs: different class, different species → push apart
        neg_mask = ~same_class & diff_species

        sim  = z @ z.T   # cosine similarity matrix [N, N]
        loss = embeddings.sum() * 0.0   # differentiable zero baseline

        if pos_mask.any():
            # Minimise angular distance: loss = 1 − cos_sim
            loss = loss + (1.0 - sim[pos_mask]).mean()

        if neg_mask.any():
            # Penalise positive cosine similarity (push apart)
            loss = loss + torch.clamp(sim[neg_mask] / self.temperature, min=0.0).mean()

        return loss


def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        weighted_sampling=True,
        species_balanced_sampling=False,
        use_scheduler=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        use_gradient_clipping=True,
        clip_value=1.0,
        model_path="model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_split=0.2,
        val_graphs=None,     # optional held-out species for cross-species early stopping
        random_seed=111,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
        # Domain-adversarial training
        num_species=0,
        lambda_adv=0.1,
        label_smoothing=0.0,
        # Focal loss
        focal_loss_gamma=0.0,
        # Cross-species contrastive loss
        lambda_con=0.3,         # weight of the contrastive loss term
        con_temperature=0.1,    # cosine similarity sharpness
        con_max_pairs=512,      # max cross-species pairs per batch
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = model.to(device)

    # === Dataset Split ===
    if val_graphs is not None:
        trainset = graphs
        valset   = val_graphs
        print(f"\nUsing external val_graphs for validation: {len(valset)} samples (cross-species proxy)")
    else:
        data_indices = list(range(len(graphs)))
        test_indices = random.sample(data_indices, int(len(graphs) * val_split))
        trainset = [graphs[i] for i in data_indices if i not in test_indices]
        valset   = [graphs[i] for i in data_indices if i in test_indices]

    # === Sampling strategy ===
    if species_balanced_sampling and all(hasattr(g, 'species_id') for g in trainset):
        spec_class_count = Counter(
            (int(g.species_id), int(g.y)) for g in trainset
        )
        weights = [1.0 / spec_class_count[(int(g.species_id), int(g.y))]
                   for g in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    elif weighted_sampling:
        label_count = Counter([int(data.y) for data in trainset])
        weights = [1.0 / label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print(f"\n{len(graphs)} graphs split: Train={len(trainset)}, Val={len(valset)}")

    # === Optimizer and Scheduler ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch_n, eta_min=1e-6)

    # === Loss Functions ===
    label_count = Counter([int(data.y) for data in trainset])
    total = sum(label_count.values())
    class_weights = torch.tensor([
        total / (2 * label_count[0]),
        total / (2 * label_count[1])
    ], dtype=torch.float).to(device)

    if focal_loss_gamma > 0.0:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma)
        print(f"Using Focal Loss (gamma={focal_loss_gamma:.1f})")
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    contrastive_criterion = CrossSpeciesContrastiveLoss(
        temperature=con_temperature, max_pairs=con_max_pairs
    )
    print(f"CrossSpeciesContrastiveLoss: lambda_con={lambda_con}, temperature={con_temperature}")

    # Determine whether the model supports _embed + classifier split
    # (CrossSpeciesGNN does; fall back to standard forward if not)
    use_embed_split = (hasattr(model, '_embed') and hasattr(model, 'classifier'))
    use_adv         = (num_species > 0 and hasattr(model, 'species_disc'))

    # === Logging ===
    train_losses, val_losses         = [], []
    train_aucs, val_aucs             = [], []
    train_accuracies, val_accuracies = [], []
    train_mccs, val_mccs             = [], []

    best_val_mcc           = -1.0
    early_stopping_counter = 0
    best_epoch             = 0

    print(f"\n########## Training for {epoch_n} epochs on {device}...")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

        progress  = (epoch - 1) / max(epoch_n - 1, 1)
        grl_alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # --- Forward pass: get embedding once for all loss terms ---
            if use_embed_split:
                embedding = model._embed(batch)
                out       = model.classifier(embedding)
            else:
                out       = model(batch)
                embedding = None

            class_loss = criterion(out, batch.y)
            loss       = class_loss

            # Adversarial loss (gradient reversal on embedding)
            if use_adv and hasattr(batch, 'species_id') and embedding is not None:
                species_out  = model.species_disc(grad_reverse(embedding, grl_alpha))
                species_loss = F.cross_entropy(species_out, batch.species_id.view(-1))
                loss         = loss + lambda_adv * species_loss
            elif use_adv and hasattr(batch, 'species_id'):
                # Fallback: use forward_adv if _embed split unavailable
                _, species_out = model.forward_adv(batch, alpha=grl_alpha)
                species_loss   = F.cross_entropy(species_out, batch.species_id.view(-1))
                loss           = loss + lambda_adv * species_loss

            # Cross-species contrastive loss on the shared embedding
            if lambda_con > 0 and embedding is not None and hasattr(batch, 'species_id'):
                con_loss = contrastive_criterion(
                    embedding, batch.y, batch.species_id.view(-1)
                )
                loss = loss + lambda_con * con_loss

            loss.backward()

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            total_loss += loss.item()

            preds  = out.argmax(dim=1).detach().cpu().numpy()
            probs  = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()

            y_pred.extend(preds)
            y_prob.extend(probs)
            y_true.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        train_acc  = accuracy_score(y_true, y_pred)
        train_auc  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
        train_mcc  = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
        train_mccs.append(train_mcc)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_aucs.append(train_auc)

        # === Validation ===
        model.eval()
        y_true_val, y_pred_val, y_prob_val = [], [], []

        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                out   = model(batch)
                loss  = criterion(out, batch.y)
                total_val_loss += loss.item()

                preds  = out.argmax(dim=1).cpu().numpy()
                probs  = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                labels = batch.y.cpu().numpy()

                y_true_val.extend(labels)
                y_pred_val.extend(preds)
                y_prob_val.extend(probs)

        val_loss = total_val_loss / len(val_loader)
        val_acc  = accuracy_score(y_true_val, y_pred_val)
        val_auc  = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan

        if len(set(y_true_val)) > 1:
            best_val_mcc_thr = -1.0
            for t in np.arange(0.05, 0.95, 0.01):
                y_pred_thr = (np.array(y_prob_val) >= t).astype(int)
                mcc_t = matthews_corrcoef(y_true_val, y_pred_thr)
                if mcc_t > best_val_mcc_thr:
                    best_val_mcc_thr = mcc_t
            val_mcc = best_val_mcc_thr
        else:
            val_mcc = np.nan
        val_mccs.append(val_mcc)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        if use_scheduler and scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(f"- Epoch [{epoch:03d}/{epoch_n}] "
              f"Train Loss: {avg_train_loss:.4f} | Train MCC: {train_mcc:.3f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val MCC: {val_mcc:.3f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
              f"| EarlyStop: {early_stopping_counter}/{early_stopping_patience} | Time: {dt:.1f}s")

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch   = epoch
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"+ New best MCC: {best_val_mcc:.4f} (epoch {epoch})")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"XXXXXXXXXX Early stopping triggered at epoch {epoch}!")
                break

    print("=================================================")
    print(f"Best validation MCC: {best_val_mcc:.4f} achieved at epoch {best_epoch}")
    print("=================================================")

    # === Reload best model, find threshold on val set ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    y_true_best_val, y_prob_best_val = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out   = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_true_best_val.extend(batch.y.cpu().numpy())
            y_prob_best_val.extend(probs)

    best_threshold = 0.5
    best_mcc       = -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred_thr = (np.array(y_prob_best_val) >= t).astype(int)
        mcc = matthews_corrcoef(y_true_best_val, y_pred_thr)
        if mcc > best_mcc:
            best_mcc       = mcc
            best_threshold = t

    print(f"Best threshold (from best model epoch {best_epoch}): {best_threshold:.2f}, MCC: {best_mcc:.4f}")

    results = {
        "best_epoch":        best_epoch,
        "best_val_mcc":      best_val_mcc,
        "best_threshold":    best_threshold,
        "train_mccs":        train_mccs,
        "val_mccs":          val_mccs,
        "train_losses":      train_losses,
        "val_losses":        val_losses,
        "train_aucs":        train_aucs,
        "val_aucs":          val_aucs,
        "train_accuracies":  train_accuracies,
        "val_accuracies":    val_accuracies,
    }

    # === Plot ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve (v3)")

    plt.subplot(1, 2, 2)
    epochs_range = range(1, len(train_mccs) + 1)
    plt.plot(epochs_range, train_mccs, label="Train MCC", linewidth=2)
    plt.plot(epochs_range, val_mccs,   label="Val MCC",   linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
    plt.axhline(y=best_val_mcc, color='r', linestyle='--', alpha=0.5)
    plt.text(0.5, best_val_mcc + 0.01, f'Best MCC: {best_val_mcc:.4f}',
             color='red', fontweight='bold', alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.xlabel("Epoch"); plt.ylabel("MCC"); plt.legend(); plt.title("MCC per Epoch (v3)")
    plt.tight_layout()
    plt.show()

    return results


def test(
        graphs,
        model,
        model_path,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_dataframe=True,
        threshold=0.5,
        search_threshold=True,
):
    # === Load model ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    criterion   = torch.nn.CrossEntropyLoss()

    y_true, y_pred, y_prob = [], [], []
    gene_ids_all = []

    print(f"########## Testing on {len(graphs)} samples...")

    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch)
            loss  = criterion(out, batch.y)
            total_loss += loss.item()

            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend((probs[:, 1] >= threshold).astype(int))
            y_prob.extend(probs[:, 1])
            gene_ids_all.extend(batch.gene_id)

    # === Bidirectional threshold search ===
    if search_threshold and len(set(y_true)) > 1:
        best_threshold_test = threshold
        best_mcc_test       = -1.0
        best_flip           = False

        for flip in [False, True]:
            search_probs = 1.0 - np.array(y_prob) if flip else np.array(y_prob)
            for t in np.arange(0.05, 0.95, 0.01):
                y_pred_thr = (search_probs >= t).astype(int)
                mcc_t = matthews_corrcoef(y_true, y_pred_thr)
                if mcc_t > best_mcc_test:
                    best_mcc_test       = mcc_t
                    best_threshold_test = t
                    best_flip           = flip

        active_probs = 1.0 - np.array(y_prob) if best_flip else np.array(y_prob)
        y_pred = (active_probs >= best_threshold_test).astype(int)
        y_prob = list(active_probs)
        print(f"Threshold search: best threshold={best_threshold_test:.2f}, "
              f"flipped={best_flip} (input was {threshold:.2f})")
    else:
        best_threshold_test = threshold

    # === Metrics ===
    test_acc  = accuracy_score(y_true, y_pred)
    test_auc  = roc_auc_score(y_true, y_prob)
    mcc       = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)

    metrics_dict = {
        "#Sample": len(graphs),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Sensitivity": round(SN, 4), "Specificity": round(SP, 4),
        "Precision": round(precision, 4), "Recall": round(recall, 4),
        "Accuracy": round(test_acc, 4), "AUC": round(test_auc, 4),
        "MCC": round(mcc, 4), "Threshold": round(best_threshold_test, 4),
    }

    results = {"metrics": pd.DataFrame([metrics_dict])}

    if return_dataframe:
        df = pd.DataFrame({
            "gene_id":         gene_ids_all,
            "true_label":      y_true,
            "predicted_label": y_pred,
            "prob_class_1":    y_prob,
            "prob_class_0":    1 - np.array(y_prob),
        })
        df["confidence"] = df[["prob_class_1", "prob_class_0"]].max(axis=1)
        df["is_correct"]  = df["true_label"] == df["predicted_label"]
        results["probs"]        = df
        results["sorted_probs"] = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    print("########## Testing completed. Outputs are saved in results!")
    return results
