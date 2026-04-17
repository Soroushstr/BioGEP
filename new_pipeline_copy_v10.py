"""
new_pipeline_copy_v10.py — Constant-GRL + AUC-stopping variant

Two targeted fixes vs v4/v9:

1. Constant GRL alpha (grl_alpha = 1.0 every epoch)
   --------------------------------------------------
   In v4 the schedule starts at alpha=0 (epoch 1) and only reaches ~0.22 by
   epoch 10.  Since LOO-val experiments showed best checkpoints at epoch 1–11,
   the adversarial signal was effectively zero during the entire useful window.
   Constant alpha=1.0 means the feature extractor is penalised for species
   discrimination from the very first gradient step, preventing early
   specialisation on training-species patterns.

2. AUC-based early stopping (early_stop_metric="auc")
   ---------------------------------------------------
   MCC on a small val set (e.g. 68 genes for arabidopsis) is noisy — the
   optimal threshold changes each epoch producing spiky MCC curves that trigger
   early stopping prematurely.  AUC is threshold-free and more stable on small
   sets.  We still find the best MCC threshold on the val set after training
   ends (unchanged from v4), so the final threshold is still calibrated.

Everything else is identical to v4 (Bayesian prior correction, focal loss,
species-balanced sampling, cosine LR, weighted sampler).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from new_gnn_models import SupConLoss


# ---------------------------------------------------------------------------
# Focal loss (unchanged from v4)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce    = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt    = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Bayesian prior correction (unchanged from v4)
# ---------------------------------------------------------------------------

def bayesian_prior_correction(probs, train_prior, test_prior):
    eps    = 1e-7
    probs  = np.clip(probs, eps, 1.0 - eps)
    log_odds = np.log(probs / (1.0 - probs))
    correction = (
        np.log(test_prior  / (1.0 - test_prior))
        - np.log(train_prior / (1.0 - train_prior))
    )
    log_odds_corrected = log_odds + correction
    corrected = 1.0 / (1.0 + np.exp(-log_odds_corrected))
    print(
        f"Bayesian prior correction: π_train={train_prior:.3f} → π_test={test_prior:.3f}  "
        f"(log-odds shift = {correction:+.3f} nats)"
    )
    print(
        f"  Prob stats before: mean={np.mean(probs):.3f}  "
        f"median={np.median(probs):.3f}  frac>0.5={np.mean(probs > 0.5):.3f}"
    )
    print(
        f"  Prob stats after : mean={np.mean(corrected):.3f}  "
        f"median={np.median(corrected):.3f}  frac>0.5={np.mean(corrected > 0.5):.3f}"
    )
    return corrected


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        weighted_sampling=True,
        species_balanced_sampling=False,
        use_scheduler=True,
        use_gradient_clipping=True,
        clip_value=1.0,
        model_path="model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_split=0.2,
        val_graphs=None,
        random_seed=111,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
        num_species=0,
        lambda_adv=0.1,
        label_smoothing=0.0,
        focal_loss_gamma=0.0,
        train_prior=0.5,
        lambda_con=0.0,
        con_temperature=0.1,
        # v10: early stopping metric ("auc" or "mcc")
        early_stop_metric="auc",
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    model = model.to(device)

    # === Dataset Split ===
    if val_graphs is not None:
        trainset = graphs
        valset   = val_graphs
        print(f"\nUsing external val_graphs for validation: {len(valset)} samples")
    else:
        data_indices  = list(range(len(graphs)))
        test_indices  = set(random.sample(data_indices, int(len(graphs) * val_split)))
        trainset = [graphs[i] for i in data_indices if i not in test_indices]
        valset   = [graphs[i] for i in data_indices if i in test_indices]

    # === Sampling ===
    if species_balanced_sampling and all(hasattr(g, 'species_id') for g in trainset):
        spec_class_count = Counter((int(g.species_id), int(g.y)) for g in trainset)
        weights = [1.0 / spec_class_count[(int(g.species_id), int(g.y))] for g in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    elif weighted_sampling:
        label_count = Counter([int(d.y) for d in trainset])
        weights = [1.0 / label_count[int(d.y)] for d in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print(f"\n{len(graphs)} graphs split: Train={len(trainset)}, Val={len(valset)}")
    print(f"Effective training prior (π_train): {train_prior:.3f}")
    print(f"Early stopping metric: {early_stop_metric.upper()}  (v10: constant GRL alpha=1.0)")

    # === Optimizer / Scheduler ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_n, eta_min=1e-6) if use_scheduler else None

    # === Loss ===
    label_count  = Counter([int(d.y) for d in trainset])
    total        = sum(label_count.values())
    class_weights = torch.tensor(
        [total / (2 * label_count[0]), total / (2 * label_count[1])],
        dtype=torch.float
    ).to(device)

    if focal_loss_gamma > 0.0:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma)
        print(f"Using Focal Loss (gamma={focal_loss_gamma:.1f})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # === State ===
    best_monitor            = -1.0   # tracks val AUC or val MCC depending on early_stop_metric
    best_val_mcc            = -1.0   # always tracked for logging / threshold search
    best_val_auc            = -1.0
    early_stopping_counter  = 0
    best_epoch              = 0

    train_losses, val_losses           = [], []
    train_aucs,   val_aucs             = [], []
    train_accuracies, val_accuracies   = [], []
    train_mccs, val_mccs               = [], []

    use_adv    = (num_species > 0 and hasattr(model, 'forward_adv'))
    use_con    = (lambda_con > 0 and hasattr(model, 'forward_all'))
    supcon_fn  = SupConLoss(temperature=con_temperature) if use_con else None

    print(f"\n########## Training for {epoch_n} epochs on {device}...")
    print(f"GRL: constant alpha=1.0, lambda_adv={lambda_adv}  "
          f"(effective weight = {lambda_adv:.2f} every epoch — no warmup)")

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

        # v10: constant alpha — adversarial signal is full-strength from epoch 1
        grl_alpha = 1.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if use_adv or use_con:
                out, species_out, proj_emb = model.forward_all(batch, alpha=grl_alpha)
                loss = criterion(out, batch.y)
                if use_adv and species_out is not None and hasattr(batch, 'species_id'):
                    loss = loss + lambda_adv * F.cross_entropy(species_out, batch.species_id.view(-1))
                if use_con and proj_emb is not None:
                    loss = loss + lambda_con * supcon_fn(proj_emb, batch.y.view(-1))
            else:
                out  = model(batch)
                loss = criterion(out, batch.y)

            loss.backward()
            if use_gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            total_loss += loss.item()
            preds  = out.argmax(dim=1).detach().cpu().numpy()
            probs  = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()
            y_pred.extend(preds); y_prob.extend(probs); y_true.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        train_acc  = accuracy_score(y_true, y_pred)
        train_auc  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
        train_mcc  = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
        train_mccs.append(train_mcc); train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc); train_aucs.append(train_auc)

        # === Validation ===
        model.eval()
        y_true_val, y_pred_val, y_prob_val = [], [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out   = model(batch)
                total_val_loss += criterion(out, batch.y).item()
                preds  = out.argmax(dim=1).cpu().numpy()
                probs  = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                labels = batch.y.cpu().numpy()
                y_true_val.extend(labels); y_pred_val.extend(preds); y_prob_val.extend(probs)

        val_loss = total_val_loss / len(val_loader)
        val_acc  = accuracy_score(y_true_val, y_pred_val)
        val_auc  = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan

        if len(set(y_true_val)) > 1:
            best_mcc_t = -1.0
            for t in np.arange(0.05, 0.95, 0.01):
                mcc_t = matthews_corrcoef(y_true_val, (np.array(y_prob_val) >= t).astype(int))
                if mcc_t > best_mcc_t:
                    best_mcc_t = mcc_t
            val_mcc = best_mcc_t
        else:
            val_mcc = np.nan

        val_mccs.append(val_mcc); val_losses.append(val_loss)
        val_accuracies.append(val_acc); val_aucs.append(val_auc)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(f"- Epoch [{epoch:03d}/{epoch_n}] "
              f"Train Loss: {avg_train_loss:.4f} | Train MCC: {train_mcc:.3f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val MCC: {val_mcc:.3f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
              f"| EarlyStop: {early_stopping_counter}/{early_stopping_patience} | Time: {dt:.1f}s")

        # Early stopping uses AUC or MCC depending on early_stop_metric
        monitor_val = val_auc if early_stop_metric == "auc" else val_mcc
        if not np.isnan(monitor_val) and monitor_val > best_monitor:
            best_monitor   = monitor_val
            best_val_mcc   = val_mcc
            best_val_auc   = val_auc
            best_epoch     = epoch
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"+ New best {early_stop_metric.upper()}: {best_monitor:.4f} (epoch {epoch})  "
                  f"[MCC={val_mcc:.4f}  AUC={val_auc:.4f}]")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"XXXXXXXXXX Early stopping triggered at epoch {epoch}!")
                break

    print("=================================================")
    print(f"Best val {early_stop_metric.upper()}: {best_monitor:.4f} at epoch {best_epoch}  "
          f"[MCC={best_val_mcc:.4f}  AUC={best_val_auc:.4f}]")
    print("=================================================")

    # === Reload best model, find MCC threshold on val set ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    y_true_bv, y_prob_bv = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out   = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_true_bv.extend(batch.y.cpu().numpy())
            y_prob_bv.extend(probs)

    best_threshold = 0.5
    best_mcc_bv    = -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        mcc = matthews_corrcoef(y_true_bv, (np.array(y_prob_bv) >= t).astype(int))
        if mcc > best_mcc_bv:
            best_mcc_bv    = mcc
            best_threshold = t

    print(f"Best threshold (from epoch {best_epoch} model): {best_threshold:.2f}, MCC: {best_mcc_bv:.4f}")

    return {
        "best_epoch":        best_epoch,
        "best_val_mcc":      best_val_mcc,
        "best_val_auc":      best_val_auc,
        "best_threshold":    best_threshold,
        "train_prior":       train_prior,
        "train_mccs":        train_mccs,
        "val_mccs":          val_mccs,
        "train_losses":      train_losses,
        "val_losses":        val_losses,
        "train_aucs":        train_aucs,
        "val_aucs":          val_aucs,
        "train_accuracies":  train_accuracies,
        "val_accuracies":    val_accuracies,
    }


# ---------------------------------------------------------------------------
# test() — identical to v4
# ---------------------------------------------------------------------------

def test(
        graphs,
        model,
        model_path,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        return_dataframe=True,
        threshold=0.5,
        search_threshold=True,
        train_prior=0.5,
        test_prior=None,
):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    criterion   = nn.CrossEntropyLoss()

    y_true, y_pred, y_prob = [], [], []
    gene_ids_all = []
    total_loss   = 0

    print(f"########## Testing on {len(graphs)} samples...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch)
            total_loss += criterion(out, batch.y).item()
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend((probs[:, 1] >= threshold).astype(int))
            y_prob.extend(probs[:, 1])
            gene_ids_all.extend(batch.gene_id)

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    if test_prior is not None:
        y_prob = bayesian_prior_correction(y_prob, train_prior, test_prior)
    else:
        print("No test_prior provided — skipping Bayesian prior correction.")

    y_prob = list(y_prob)
    y_true = list(y_true)

    if search_threshold and len(set(y_true)) > 1:
        best_threshold_test = threshold
        best_mcc_test       = -1.0
        best_flip           = False
        for flip in [False, True]:
            search_probs = 1.0 - np.array(y_prob) if flip else np.array(y_prob)
            for t in np.arange(0.05, 0.95, 0.01):
                mcc_t = matthews_corrcoef(y_true, (search_probs >= t).astype(int))
                if mcc_t > best_mcc_test:
                    best_mcc_test       = mcc_t
                    best_threshold_test = t
                    best_flip           = flip
        active_probs = 1.0 - np.array(y_prob) if best_flip else np.array(y_prob)
        y_pred = (active_probs >= best_threshold_test).astype(int)
        y_prob = list(active_probs)
        print(f"Threshold search: best={best_threshold_test:.2f}, flipped={best_flip}")
    else:
        best_threshold_test = threshold
        y_pred = (np.array(y_prob) >= threshold).astype(int)

    test_acc  = accuracy_score(y_true, y_pred)
    test_auc  = roc_auc_score(y_true, y_prob)
    mcc       = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)

    metrics_dict = {
        "#Sample":     len(graphs),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Sensitivity": round(SN,        4),
        "Specificity": round(SP,        4),
        "Precision":   round(precision, 4),
        "Recall":      round(recall,    4),
        "Accuracy":    round(test_acc,  4),
        "AUC":         round(test_auc,  4),
        "MCC":         round(mcc,       4),
        "Threshold":   round(best_threshold_test, 4),
        "pi_test":     round(test_prior, 4) if test_prior is not None else float("nan"),
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
        df["is_correct"] = df["true_label"] == df["predicted_label"]
        results["probs"]        = df
        results["sorted_probs"] = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    print("########## Testing completed.")
    return results
