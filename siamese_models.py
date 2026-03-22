import torch
import torch.nn as nn

from siamese_encoders import CNN1DEncoder, CNNBagEncoder


class SiameseCNNPairClassifier(nn.Module):
    def __init__(
        self,
        input_len,
        emb_dim=32,
        hidden=64,
        encoder_dropout=0.2,
        head_dropout=0.15,
    ):
        super().__init__()
        self.encoder = CNN1DEncoder(
            input_len=input_len,
            emb_dim=emb_dim,
            dropout=encoder_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        diff = torch.abs(z1 - z2)
        logits = self.head(diff)
        return logits


class SiameseBagPairClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim=16,
        c1=16,
        c2=32,
        kernel_size=3,
        encoder_dropout=0.2,
        pooling="mean",
        head_hidden=32,
        head_dropout=0.5,
    ):
        super().__init__()

        self.encoder = CNNBagEncoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            c1=c1,
            c2=c2,
            kernel_size=kernel_size,
            dropout=encoder_dropout,
            pooling=pooling,
        )

        self.head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, bag1, bag2):
        z1 = self.encoder(bag1)
        z2 = self.encoder(bag2)
        diff = torch.abs(z1 - z2)
        logits = self.head(diff)
        return logits