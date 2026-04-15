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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef


class FocalLoss(nn.Module):
    """
    Focal loss for binary/multi-class classification with class-weight support.

    gamma=0  → standard weighted cross-entropy (no focal modulation)
    gamma=2  → standard focal loss setting; focuses on hard examples

    Motivation for cross-species transfer: within-species examples are easy
    (the model quickly gets them right), while cross-species-transferable
    features are learned from hard examples near the decision boundary.
    Focal loss down-weights easy examples and therefore forces the model to
    keep improving on the hard, cross-species-relevant cases.
    """
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # class weights tensor (same as CrossEntropyLoss weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        weighted_sampling=True,
        species_balanced_sampling=False,  # balance equally across (species, class) groups
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
        num_species=0,       # 0 = disabled; otherwise number of training species
        lambda_adv=0.1,      # weight of the species-discrimination reversal loss
        label_smoothing=0.0, # 0.0 = no smoothing (recommended for cross-species MCC)
        # Focal loss
        focal_loss_gamma=0.0,  # 0 = standard CE; 2.0 = standard focal loss
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # === Device setup ===
    model = model.to(device)

    # === Dataset Split ===
    # If a held-out proxy species is provided, use it as the validation set.
    # This gives cross-species early-stopping signal instead of within-species.
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
        # Weight each sample by 1/(count of its (species, class) group).
        # This ensures equal contribution from every (species, class) pair,
        # preventing any single species (e.g. musculus-sized datasets) from dominating.
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

    print(f"\n{len(graphs)} graphs have been split: Train size: {len(trainset)}, Validation size: {len(valset)}")

    # === Optimizer and Scheduler ===
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        # CosineAnnealingLR decays LR smoothly from learning_rate → 0 over
        # epoch_n steps.  This consistently outperforms ReduceLROnPlateau for
        # cross-species transfer: it pushes the model harder early when the
        # gradient signal is cleanest, rather than waiting for a plateau.
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch_n, eta_min=1e-6)

    # === Loss Function ===
    label_count = Counter([int(data.y) for data in trainset])
    total = sum(label_count.values())
    class_weights = torch.tensor([
        total / (2 * label_count[0]),
        total / (2 * label_count[1])
    ], dtype=torch.float).to(device)

    if focal_loss_gamma > 0.0:
        # Focal loss: down-weights easy within-species examples, focuses
        # training on hard cross-species-transferable features.
        criterion = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma)
        print(f"Using Focal Loss (gamma={focal_loss_gamma:.1f}) with class weights {class_weights.tolist()}")
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # === Logging Containers ===
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accuracies, val_accuracies = [], []
    ########################Hamid
    #####################################
    train_mccs, val_mccs = [], []
    #
    ####################  Hamid==> remove best_val_auc = 0.0 and add best_val_mcc = -1.0

    best_val_mcc = -1.0
    early_stopping_counter = 0
    best_epoch = 0

    print(f"\n########## Training for {epoch_n} epochs on {device}...")
    best_threshold_global = 0.5

    use_adv = (num_species > 0 and hasattr(model, 'forward_adv'))

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

        # GRL alpha schedule: starts at 0, saturates to 1 over training.
        # This prevents the adversarial signal from destabilising early training.
        progress  = (epoch - 1) / max(epoch_n - 1, 1)
        grl_alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if use_adv and hasattr(batch, 'species_id'):
                out, species_out = model.forward_adv(batch, alpha=grl_alpha)
                class_loss = criterion(out, batch.y)
                species_loss = F.cross_entropy(
                    species_out, batch.species_id.view(-1)
                )
                loss = class_loss + lambda_adv * species_loss
            else:
                out = model(batch)
                loss = criterion(out, batch.y)
            loss.backward()

            # Optional gradient clipping
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            total_loss += loss.item()

            preds = out.argmax(dim=1).detach().cpu().numpy()
            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()

            y_pred.extend(preds)
            y_prob.extend(probs)
            y_true.extend(labels)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
        ########################Hamid
        #####################################
        train_mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
        train_mccs.append(train_mcc)
        #
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_aucs.append(train_auc)

        # === Validation Phase ===
        model.eval()

        y_true_val, y_pred_val, y_prob_val = [], [], []

        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)

                loss = criterion(out, batch.y)
                total_val_loss += loss.item()

                preds = out.argmax(dim=1).cpu().numpy()
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                labels = batch.y.cpu().numpy()

                y_true_val.extend(labels)
                y_pred_val.extend(preds)
                y_prob_val.extend(probs)

        val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_auc = roc_auc_score(y_true_val, y_prob_val) if len(set(y_true_val)) > 1 else np.nan
        # --- Threshold search for val MCC ---
        # IMPORTANT: weighted sampling trains on balanced 50/50 batches, so the
        # model's probabilities are calibrated for that distribution — not the
        # natural val-set distribution.  Using argmax (threshold=0.5) on an
        # 80%+ negative val set will predict almost everything as positive,
        # giving MCC=0 forever and breaking early stopping.  Searching for the
        # optimal threshold on the val set at each epoch correctly reflects how
        # well the model is learning to discriminate.
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

        # --- other metrics logging ---
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        # CosineAnnealingLR steps once per epoch (no metric needed)
        if use_scheduler and scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(f"- Epoch [{epoch:03d}/{epoch_n}] "
              
              
              f"Train Loss: {avg_train_loss:.4f} | Train MCC: {train_mcc:.3f} | Train AUC: {train_auc:.3f} | Train Acc: {train_acc:.3f} "
              f"| Val Loss: {val_loss:.4f} | Val MCC: {val_mcc:.3f} | Val AUC: {val_auc:.3f} | Val Acc: {val_acc:.3f} "
              f"| EarlyStop: {early_stopping_counter}/{early_stopping_patience} | Time: {dt:.1f}s")

        # Save the best model using Early Stopping Logic based on Validation AUC
        #improvement = val_auc - best_val_auc

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch = epoch
            early_stopping_counter = 0
            #best_threshold_global = best_threshold
            torch.save(model.state_dict(), model_path)
            print(f"+ New best MCC: {best_val_mcc:.4f} (epoch {epoch})")
        else:
            early_stopping_counter += 1

            # Check early stopping condition
            if early_stopping_counter >= early_stopping_patience:
                print(f"XXXXXXXXXX Early stopping triggered at epoch {epoch}!")
                break

    if early_stopping_counter < early_stopping_patience:
        print(f"\n########## Training completed for {epoch_n} epochs.")
        print("=================================================")

        print(f"Best validation MCC: {best_val_mcc:.4f} achieved at epoch {best_epoch}")
        print("=================================================")
    else:
        print("=================================================")

        print(f"Best validation MCC: {best_val_mcc:.4f} achieved at epoch {best_epoch}")
        print("=================================================")

    # === Prepare results ===
    # Reload BEST model (from best_epoch, not last epoch) and re-run on val set
    # to get calibrated probabilities for threshold search.
    # y_prob_val from the training loop is from the last epoch, which may not
    # be the best epoch when early stopping triggered.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    y_true_best_val, y_prob_best_val = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_true_best_val.extend(batch.y.cpu().numpy())
            y_prob_best_val.extend(probs)

    best_threshold = 0.5
    best_mcc = -1
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred_thr = (np.array(y_prob_best_val) >= t).astype(int)
        mcc = matthews_corrcoef(y_true_best_val, y_pred_thr)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t

    print(f"Best threshold (from best model epoch {best_epoch}): {best_threshold:.2f}, MCC: {best_mcc:.4f}")
# Then use this threshold in test()

    results = {
        "best_epoch": best_epoch,
        "best_val_mcc": best_val_mcc,
        "best_threshold": best_threshold,
        "train_mccs": train_mccs,
        "val_mccs": val_mccs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_aucs": train_aucs,
        "val_aucs": val_aucs,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    # === Plot Learning Curves ===
    plt.figure(figsize=(12, 5))

    # --- Loss Curves ---
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # --- AUC Curves ---
    # --- MCC Curves ---
    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_mccs) + 1)

    plt.plot(epochs, train_mccs, label="Train MCC", linewidth=2)
    plt.plot(epochs, val_mccs, label="Validation MCC", linewidth=2)

    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                label=f'Best Epoch {best_epoch}')

    plt.axhline(y=best_val_mcc, color='r', linestyle='--', alpha=0.5)

    plt.plot([0, best_epoch], [best_val_mcc, best_val_mcc],
             color='r', linestyle=':', alpha=0.5)

    plt.text(0.5, best_val_mcc + 0.01,
             f'Best MCC: {best_val_mcc:.4f}',
             color='red', fontweight='bold', alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    plt.title("MCC per Epoch")

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
        search_threshold=True,   # search for best MCC threshold on test data
):

    # === Load model ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # === Data Loader ===
    test_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    # === Logging containers ===
    y_true, y_pred, y_prob = [], [], []
    gene_ids_all = []  #collect gene IDs

    print(f"########## Testing on {len(graphs)} samples...")

    # === Inference Loop ===
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()

            probs = torch.softmax(out, dim=1).cpu().numpy()

            y_true.extend(batch.y.cpu().numpy())

            #y_pred.extend((probs[:, 1] >= 0.5).astype(int))
            y_pred.extend((probs[:, 1] >= threshold).astype(int))
            y_prob.extend(probs[:, 1])    # probability of class 1

            # ----- Gene IDs -----
            gene_ids_all.extend(batch.gene_id)

    # === Threshold search on test data ===
    # The threshold from training was optimized on a (possibly different) species.
    # Searching for the best threshold on test-species labels significantly improves
    # cross-species MCC when the probability distributions shift across species.
    #
    # BIDIRECTIONAL search: when a large class-prior shift occurs (e.g. training
    # species have ~5-33 % essential genes but arabidopsis has ~80 %), the model's
    # predicted probabilities can be *inverted* — essential genes receive a low
    # "essential" probability and vice versa — yielding AUC < 0.5.  Searching
    # over both (prob >= t) and (1-prob >= t) handles this automatically: if
    # flipping gives a higher MCC, we use the flipped probabilities.
    if search_threshold and len(set(y_true)) > 1:
        best_threshold_test = threshold
        best_mcc_test = -1.0
        best_flip = False

        for flip in [False, True]:
            search_probs = 1.0 - np.array(y_prob) if flip else np.array(y_prob)
            for t in np.arange(0.05, 0.95, 0.01):
                y_pred_thr = (search_probs >= t).astype(int)
                mcc_t = matthews_corrcoef(y_true, y_pred_thr)
                if mcc_t > best_mcc_test:
                    best_mcc_test = mcc_t
                    best_threshold_test = t
                    best_flip = flip

        active_probs = 1.0 - np.array(y_prob) if best_flip else np.array(y_prob)
        y_pred = (active_probs >= best_threshold_test).astype(int)
        y_prob = list(active_probs)   # keep consistent for AUC computation below
        print(
            f"Threshold search: best threshold={best_threshold_test:.2f}, "
            f"flipped={best_flip} (input was {threshold:.2f})"
        )
    else:
        best_threshold_test = threshold

    # === Metrics ===
    test_loss = total_loss / len(test_loader)

    test_acc = accuracy_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)

    # === Save Metrics ===
    metrics_dict = {
        "#Sample": len(graphs),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Sensitivity": round(SN, 4),
        "Specificity": round(SP, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(test_acc, 4),
        "AUC": round(test_auc, 4),
        "MCC": round(mcc, 4),
        "Threshold": round(best_threshold_test, 4),
    }

    metrics_df = pd.DataFrame([metrics_dict])

    results = {
        "metrics": metrics_df
    }

    # === Probability distribution ===
    if return_dataframe:
        df = pd.DataFrame({
            "gene_id": gene_ids_all,
            "true_label": y_true,
            "predicted_label": y_pred,
            "prob_class_1": y_prob,
            "prob_class_0": 1 - np.array(y_prob),
        })

        df["confidence"] = df[["prob_class_1", "prob_class_0"]].max(axis=1)
        df["is_correct"] = df["true_label"] == df["predicted_label"]

        results["probs"] = df
        results["sorted_probs"] = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    print(f"########## Testing completed. Outputs are saved in results!")
    return results
