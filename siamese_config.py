import torch

SEED = 42
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PAIR_CONFIG = {
    "seed": SEED,
    "batch_size": 512,
    "lr": 1e-3,
    "max_epochs": 30,
    "patience": 6,
    "train_pairs": 200000,
    "val_pairs": 60000,
    "test_pairs": 80000,
    "pos_fraction": 0.5,
    "include_ba": True,
}

BLOCK_CONFIG = {
    "seed": SEED,
    "block_size": 5,
    "train_pairs": 30000,
    "val_pairs": 20000,
    "test_pairs": 30000,
    "pos_fraction": 0.5,
    "include_ba": True,
    "within_block_replace": False,
    "max_row_reuse": 20,

    "batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 1e-2,
    "max_epochs": 30,
    "patience": 6,

    "emb_dim": 8,
    "c1": 8,
    "c2": 16,
    "kernel_size": 3,
    "encoder_dropout": 0.20,
    "pooling": "mean",

    "head_hidden": 12,
    "head_dropout": 0.35,
}