import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, accuracy_score

from siamese_eval import predict_pair_scores, predict_scores_and_loss, compute_metrics


def train_pair_classifier(
    model,
    train_loader,
    train_eval_loader,
    val_loader,
    device,
    lr=1e-3,
    max_epochs=30,
    patience=6,
):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_auc = -np.inf
    best_state = None
    bad = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x1, x2, y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x1, x2)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        train_scores, train_labels = predict_pair_scores(model, train_eval_loader, device)
        train_auc = float(roc_auc_score(train_labels, train_scores))
        train_acc = float(accuracy_score(train_labels, (train_scores >= 0.5).astype(int)))

        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        val_probs = []
        val_lbls = []

        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                logits = model(x1, x2)
                vloss = crit(logits, y)

                val_running_loss += vloss.item()
                val_batches += 1

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.append(probs)
                val_lbls.append(y.cpu().numpy())

        val_scores = np.concatenate(val_probs)
        val_labels = np.concatenate(val_lbls)
        val_loss = val_running_loss / max(val_batches, 1)

        val_auc = float(roc_auc_score(val_labels, val_scores))
        val_acc = float(accuracy_score(val_labels, (val_scores >= 0.5).astype(int)))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train AUC={train_auc:.4f} acc={train_acc:.4f} | "
            f"val AUC={val_auc:.4f} acc={val_acc:.4f}"
        )

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_bag_classifier(
    model,
    train_loader,
    val_loader,
    device,
    lr=5e-4,
    weight_decay=1e-2,
    max_epochs=30,
    patience=6,
):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_val_auc = -np.inf
    best_state = None
    bad = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for bag1, bag2, y in train_loader:
            bag1 = bag1.to(device)
            bag2 = bag2.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(bag1, bag2)
            loss = crit(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        train_scores, train_labels, _ = predict_scores_and_loss(model, train_loader, device)
        train_auc, train_acc, _, _ = compute_metrics(train_scores, train_labels)

        val_scores, val_labels, val_loss = predict_scores_and_loss(model, val_loader, device)
        val_auc, val_acc, _, _ = compute_metrics(val_scores, val_labels)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train AUC={train_auc:.4f} acc={train_acc:.4f} | "
            f"val AUC={val_auc:.4f} acc={val_acc:.4f}"
        )

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history