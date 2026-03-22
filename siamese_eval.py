import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


@torch.no_grad()
def predict_pair_scores(model, loader, device):
    model.eval()
    probs = []
    labels = []

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        logits = model(x1, x2)
        p_diff = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        probs.append(p_diff)
        labels.append(y.numpy())

    return np.concatenate(probs), np.concatenate(labels)


def evaluate_pairs(scores, labels, threshold=0.5):
    auc = float(roc_auc_score(labels, scores))
    preds = (scores >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds))
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc, acc, fpr, tpr


@torch.no_grad()
def predict_scores_and_loss(model, loader, device):
    model.eval()
    crit = torch.nn.CrossEntropyLoss()

    all_scores = []
    all_labels = []
    loss_sum = 0.0
    n_batches = 0

    for bag1, bag2, y in loader:
        bag1 = bag1.to(device)
        bag2 = bag2.to(device)
        y = y.to(device)

        logits = model(bag1, bag2)
        loss = crit(logits, y)

        loss_sum += loss.item()
        n_batches += 1

        scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels = y.detach().cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    avg_loss = loss_sum / max(n_batches, 1)

    return scores, labels, avg_loss


def compute_metrics(scores, labels, threshold=0.5):
    auc = float(roc_auc_score(labels, scores))
    preds = (scores >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds))
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc, acc, fpr, tpr


def plot_roc_curve(fpr, tpr, auc, title="ROC Curve"):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_binary_roc_from_scores(y_true, scores, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plot_roc_curve(fpr, tpr, auc, title=title)