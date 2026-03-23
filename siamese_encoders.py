import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DEncoder(nn.Module):
    def __init__(self, input_len, emb_dim=16, c1=16, c2=32, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(1, c1, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            h = self._forward_conv(dummy)
            flat_dim = h.view(1, -1).shape[1]

        self.fc = nn.Linear(flat_dim, emb_dim)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self._forward_conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class CNNBlockEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim=16,
        c1=16,
        c2=32,
        kernel_size=3,
        dropout=0.2,
        pooling="mean",
        l2_normalize=True,
    ):
        super().__init__()

        self.row_encoder = CNN1DEncoder(
            input_len=input_dim,
            emb_dim=emb_dim,
            c1=c1,
            c2=c2,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be 'mean' or 'max'")

        self.pooling = pooling
        self.l2_normalize = l2_normalize

    def forward(self, block_x):
        B, K, D = block_x.shape

        x_flat = block_x.view(B * K, D)         # (B*K, D)
        row_emb = self.row_encoder(x_flat)      # (B*K, emb_dim)
        row_emb = row_emb.view(B, K, -1)        # (B, K, emb_dim)

        if self.pooling == "mean":
            z = row_emb.mean(dim=1)
        else:
            z, _ = row_emb.max(dim=1)

        if self.l2_normalize:
            z = F.normalize(z, p=2, dim=1)

        return z


class BlockEncoder(nn.Module):
    def __init__(self, input_dim, row_hidden=(32, 32), emb_dim=16, dropout=0.35):
        super().__init__()
        layers = []
        prev = input_dim
        for h in row_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.row_net = nn.Sequential(*layers)
        self.to_emb = nn.Linear(prev, emb_dim)

    def forward(self, block_x):
        B, K, D = block_x.shape
        x_flat = block_x.view(B * K, D)
        h = self.row_net(x_flat)
        h = h.view(B, K, -1)
        pooled = h.mean(dim=1)
        z = self.to_emb(pooled)
        z = F.normalize(z, p=2, dim=1)
        return z