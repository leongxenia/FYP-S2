import numpy as np
import torch
import torch.nn as nn

def pairwise_distances(emb):
    dot = emb @ emb.t()
    sq = torch.sum(emb ** 2, dim=1, keepdim=True)
    dist2 = sq - 2 * dot + sq.t()
    dist2 = torch.clamp(dist2, min=0.0)
    return torch.sqrt(dist2 + 1e-12)


def select_pairs_hard_negative(labels, emb, n_neg_per_pos=1):
    labels = labels.detach().cpu().numpy()
    dist = pairwise_distances(emb).detach()

    idx_by_class = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(int(y), []).append(i)

    pos_pairs = []
    neg_pairs = []

    classes = list(idx_by_class.keys())
    if len(classes) < 2:
        return (
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
        )

    for c in classes:
        idxs = idx_by_class[c]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pos_pairs.append([idxs[a], idxs[b]])

    for c in classes:
        idxs_pos = idx_by_class[c]
        idxs_neg = []
        for c2 in classes:
            if c2 != c:
                idxs_neg.extend(idx_by_class[c2])

        if len(idxs_neg) == 0:
            continue

        for i in idxs_pos:
            drow = dist[i, idxs_neg]
            hard_idx = torch.argsort(drow)[:n_neg_per_pos].cpu().numpy()
            for h in hard_idx:
                neg_pairs.append([i, idxs_neg[int(h)]])

    pos_pairs = (
        torch.tensor(pos_pairs, dtype=torch.long)
        if len(pos_pairs)
        else torch.empty((0, 2), dtype=torch.long)
    )
    neg_pairs = (
        torch.tensor(neg_pairs, dtype=torch.long)
        if len(neg_pairs)
        else torch.empty((0, 2), dtype=torch.long)
    )
    return pos_pairs, neg_pairs


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, y):
        d = torch.norm(emb1 - emb2, dim=1)
        loss_pos = (1 - y) * d.pow(2)
        loss_neg = y * torch.clamp(self.margin - d, min=0.0).pow(2)
        return torch.mean(loss_pos + loss_neg)