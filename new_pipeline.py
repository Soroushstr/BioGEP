import torch
import random
import numpy as np
import pandas as pd
import time
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
def train(
        graphs,
        model,
        batch_size=64,
        epoch_n=50,
        learning_rate=1e-3,
        weighted_sampling=True,
        use_scheduler=True,
        scheduler_patience=10,
        scheduler_factor=0.5,
        use_gradient_clipping=True,
        clip_value=1.0,
        model_path="model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_split=0.2,
        random_seed=111,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # === Device setup ===
    model = model.to(device)

    # === Dataset Split ===
    data_indices = list(range(len(graphs)))
    test_indices = random.sample(data_indices, int(len(graphs) * val_split))
    trainset = [graphs[i] for i in data_indices if i not in test_indices]
    valset = [graphs[i] for i in data_indices if i in test_indices]

    # === Weighted Sampling (for class imbalance) ===
    if weighted_sampling:
        #label_count = Counter([int(data.y) for data in graphs])
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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor)

    # === Loss Function ===
    label_count = Counter([int(data.y) for data in trainset])
    total = sum(label_count.values())
    class_weights = torch.tensor([
        total / (2 * label_count[0]),
        total / (2 * label_count[1])
    ], dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

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

    for epoch in range(1, epoch_n + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

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
        ########################Hamid
        #####################################
        # --- Threshold search (FIRST) ---
        #best_threshold = 0.5
        #best_val_mcc_thr = -1

        #for t in np.arange(0.05, 0.95, 0.01):
        #    y_pred_thr = (np.array(y_prob_val) >= t).astype(int)
        #    mcc = matthews_corrcoef(y_true_val, y_pred_thr)
        #
        #    if mcc > best_val_mcc_thr:
        #        best_val_mcc_thr = mcc
        #        best_threshold = t

        # --- MCC (BASED ON BEST THRESHOLD RESULT) ---
        #val_mcc = best_val_mcc_thr if len(set(y_true_val)) > 1 else np.nan
        # Replace your threshold search block with simply:
        val_mcc = matthews_corrcoef(y_true_val, y_pred_val)  # use argmax predictions
        val_mccs.append(val_mcc)

        # --- other metrics logging ---
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        # Scheduler Logic based on Validation AUC

        pseudo_loss = 1.0 - val_mcc if not np.isnan(val_mcc) else 1.0
        if use_scheduler and scheduler is not None:
            scheduler.step(pseudo_loss)

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
    # After training, load best model and run on val set
    # Then search for best threshold once
    best_threshold = 0.5
    best_mcc = -1

    for t in np.arange(0.05, 0.95, 0.01):
        y_pred_thr = (np.array(y_prob_val) >= t).astype(int)
        mcc = matthews_corrcoef(y_true_val, y_pred_thr)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t

    print(f"Best threshold: {best_threshold:.2f}, MCC: {best_mcc:.4f}")
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
        threshold=0.5
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
