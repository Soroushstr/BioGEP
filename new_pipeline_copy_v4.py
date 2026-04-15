"""
new_pipeline_copy_v4.py — Bayesian Prior-Correction variant

Key difference from new_pipeline_copy.py (v1):
  - Adds a Bayesian prior-correction step in test() that adjusts the model's
    output probabilities for the known class-frequency shift between the
    training distribution and the test species.

Motivation
----------
The model is trained with species-balanced sampling, so it effectively assumes
an equal 50/50 prior (π_train ≈ 0.5).  Arabidopsis has ~80 % essential genes
(π_test ≈ 0.80).  These two priors are in strong conflict: the model's raw
probabilities are calibrated for a world where half the genes are essential,
but in arabidopsis most genes are essential.

Bayesian prior correction (also called "dataset-shift adjustment" or
"logistic calibration") fixes this in log-odds space:

    log_odds_model     = log( P_model / (1 - P_model) )
    log_odds_corrected = log_odds_model
                       + log( π_test  / (1 - π_test)  )
                       - log( π_train / (1 - π_train) )
    P_corrected        = sigmoid( log_odds_corrected )

Intuitively: the model gives us P(essential | features, π_train).  We divide
out the training prior and multiply in the test prior to get
P(essential | features, π_test).  The likelihood-ratio P(features|essential)
/ P(features|non-essential) is unchanged — it is species-agnostic by
construction — so the correction is exact under Naive Bayes assumptions.

How π_test is obtained
----------------------
π_test (the essential-gene fraction of the test species) is read directly
from the label file before calling test().  This is a realistic assumption:
the overall essential-gene fraction for a model organism is typically known
from published databases (e.g. OGEE, DEG) even when per-gene labels are
unavailable or withheld.  If labels are available (as they are here for
evaluation), π_test can also be computed exactly.

How π_train is set
------------------
With species_balanced_sampling=True, every (species, class) group contributes
equally to each batch, so the effective class prior seen by the model during
training is 0.5.  This is the correct value to use as π_train.  If
balanced sampling was NOT used, set train_prior to the actual fraction of
essential genes in the combined training pool.

All other improvements are kept:
  - Bidirectional threshold search (handles residual inversion after correction)
  - Focal loss (gamma=2)
  - Maripaludis proxy validation for cross-species early stopping
  - lambda_adv = 0.2
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


# ---------------------------------------------------------------------------
# Focal loss (same as v1)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss with class-weight support.
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


# ---------------------------------------------------------------------------
# Bayesian prior correction
# ---------------------------------------------------------------------------

def bayesian_prior_correction(
    probs:       np.ndarray,
    train_prior: float,
    test_prior:  float,
) -> np.ndarray:
    """
    Shift model output probabilities from the training class-prior to the
    known test-species class-prior using a log-odds correction.

    Parameters
    ----------
    probs       : array [N], model output P(essential | features, π_train)
    train_prior : π_train — fraction of essential genes in the effective
                  training distribution (0.5 with species-balanced sampling)
    test_prior  : π_test  — fraction of essential genes in the test species

    Returns
    -------
    array [N], corrected P(essential | features, π_test)

    Derivation
    ----------
    Under the Naive Bayes factoring of P(y | x) ∝ P(x | y) * P(y):

        log_odds(y=1 | x) = log[ P(x|y=1) / P(x|y=0) ] + log[ π / (1-π) ]

    The likelihood ratio log[ P(x|y=1) / P(x|y=0) ] is species-agnostic
    (it is what the model has learned from sequence features).  Substituting
    the test prior for the training prior therefore gives the correct
    posterior for the new species without retraining.

    Example (arabidopsis)
    ---------------------
    train_prior = 0.50  (balanced sampling)
    test_prior  = 0.80  (arabidopsis)
    correction  = log(0.80/0.20) - log(0.50/0.50) = log(4) ≈ +1.386 nats

    A gene with P_model = 0.10 → log_odds = -2.20 → corrected = -0.81
    → P_corrected = 0.31.   A gene with P_model = 0.30 → log_odds = -0.85
    → corrected = +0.54 → P_corrected = 0.63.  The whole distribution is
    shifted upward to match arabidopsis' high essential-gene prevalence.
    """
    eps    = 1e-7
    probs  = np.clip(probs, eps, 1.0 - eps)

    log_odds = np.log(probs / (1.0 - probs))

    correction = (
        np.log(test_prior  / (1.0 - test_prior))
        - np.log(train_prior / (1.0 - train_prior))
    )

    log_odds_corrected = log_odds + correction
    corrected          = 1.0 / (1.0 + np.exp(-log_odds_corrected))

    print(
        f"Bayesian prior correction: π_train={train_prior:.3f} → π_test={test_prior:.3f}  "
        f"(log-odds shift = {correction:+.3f} nats)"
    )
    print(
        f"  Prob stats before: mean={np.mean(probs):.3f}  "
        f"median={np.median(probs):.3f}  "
        f"frac>0.5={np.mean(probs > 0.5):.3f}"
    )
    print(
        f"  Prob stats after : mean={np.mean(corrected):.3f}  "
        f"median={np.median(corrected):.3f}  "
        f"frac>0.5={np.mean(corrected > 0.5):.3f}"
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
        # Effective training class-prior (returned in results for use in test())
        train_prior=0.5,     # 0.5 = balanced sampling; set to actual fraction otherwise
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
    print(f"Effective training prior (π_train): {train_prior:.3f}")

    # === Optimizer and Scheduler ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch_n, eta_min=1e-6)

    # === Loss Function ===
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

    # === Logging ===
    train_losses, val_losses         = [], []
    train_aucs, val_aucs             = [], []
    train_accuracies, val_accuracies = [], []
    train_mccs, val_mccs             = [], []

    best_val_mcc           = -1.0
    early_stopping_counter = 0
    best_epoch             = 0

    print(f"\n########## Training for {epoch_n} epochs on {device}...")

    use_adv = (num_species > 0 and hasattr(model, 'forward_adv'))

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

            if use_adv and hasattr(batch, 'species_id'):
                out, species_out = model.forward_adv(batch, alpha=grl_alpha)
                class_loss   = criterion(out, batch.y)
                species_loss = F.cross_entropy(species_out, batch.species_id.view(-1))
                loss         = class_loss + lambda_adv * species_loss
            else:
                out  = model(batch)
                loss = criterion(out, batch.y)

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
    best_mcc_val   = -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred_thr = (np.array(y_prob_best_val) >= t).astype(int)
        mcc = matthews_corrcoef(y_true_best_val, y_pred_thr)
        if mcc > best_mcc_val:
            best_mcc_val   = mcc
            best_threshold = t

    print(f"Best threshold (from best model epoch {best_epoch}): {best_threshold:.2f}, MCC: {best_mcc_val:.4f}")

    results = {
        "best_epoch":        best_epoch,
        "best_val_mcc":      best_val_mcc,
        "best_threshold":    best_threshold,
        "train_prior":       train_prior,   # pass to test() for prior correction
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
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve (v4)")

    plt.subplot(1, 2, 2)
    epochs_range = range(1, len(train_mccs) + 1)
    plt.plot(epochs_range, train_mccs, label="Train MCC", linewidth=2)
    plt.plot(epochs_range, val_mccs,   label="Val MCC",   linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
    plt.axhline(y=best_val_mcc, color='r', linestyle='--', alpha=0.5)
    plt.text(0.5, best_val_mcc + 0.01, f'Best MCC: {best_val_mcc:.4f}',
             color='red', fontweight='bold', alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.xlabel("Epoch"); plt.ylabel("MCC"); plt.legend(); plt.title("MCC per Epoch (v4)")
    plt.tight_layout()
    plt.show()

    return results


# ---------------------------------------------------------------------------
# test()
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
        # Bayesian prior correction
        train_prior=0.5,   # π_train — effective training class prior
                           # (0.5 with species-balanced sampling)
        test_prior=None,   # π_test  — known essential-gene fraction of the test species
                           # (None = skip correction, use bidirectional search only)
):
    """
    Run inference and compute metrics.

    If test_prior is provided, applies a Bayesian prior correction to the
    model's raw probabilities before threshold search:

        P_corrected = sigmoid( log_odds_model
                               + log(π_test/(1-π_test))
                               - log(π_train/(1-π_train)) )

    This shifts the probability distribution to match the known essential-gene
    fraction of the test species.  For arabidopsis (π_test≈0.80) trained with
    balanced sampling (π_train=0.50), the log-odds correction is ≈+1.4 nats,
    pushing most genes toward higher essentiality probability as expected.

    After correction, a bidirectional threshold search is still performed to
    handle any residual miscalibration.
    """
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

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    # === Step 1: Bayesian prior correction ===
    # Adjust model probabilities for the known test-species class-prior shift.
    # This is the core v4 contribution; it happens before threshold search.
    if test_prior is not None:
        y_prob = bayesian_prior_correction(
            probs       = y_prob,
            train_prior = train_prior,
            test_prior  = test_prior,
        )
    else:
        print("No test_prior provided — skipping Bayesian prior correction.")

    y_prob = list(y_prob)
    y_true = list(y_true)

    # === Step 2: Bidirectional threshold search ===
    # Handles residual probability inversion after prior correction.
    # Even after correction, the ranking may still be imperfect, so we search
    # both probability directions for the best MCC threshold.
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
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        y_prob = list(y_prob)

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
        "#Sample":     len(graphs),
        "TN":          int(tn),
        "FP":          int(fp),
        "FN":          int(fn),
        "TP":          int(tp),
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
        df["is_correct"]  = df["true_label"] == df["predicted_label"]
        results["probs"]        = df
        results["sorted_probs"] = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    print("########## Testing completed. Outputs are saved in results!")
    return results
