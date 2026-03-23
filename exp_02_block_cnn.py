import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from siamese_config import BLOCK_CONFIG
from siamese_utils import set_seed, get_device
from siamese_data import prepare_encoded_data, BlockPairDataset
from siamese_models import SiameseBlockPairClassifier
from siamese_trainers import train_block_classifier
from siamese_eval import predict_scores_and_loss, compute_metrics, plot_roc_curve


def plot_history(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Block CNN: Epoch vs Loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_auc"], label="train_auc")
    plt.plot(history["epoch"], history["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Block CNN: Epoch vs AUC")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def run_experiment(preprocess, X_train, X_val, X_test, T_train, T_val, T_test):
    cfg = BLOCK_CONFIG
    set_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    Xtr, Xva, Xte, Ttr, Tva, Tte = prepare_encoded_data(
        preprocess, X_train, X_val, X_test, T_train, T_val, T_test
    )
    print("Encoded shapes:", Xtr.shape, Xva.shape, Xte.shape)

    input_dim = Xtr.shape[1]

    train_ds = BlockPairDataset(
        Xtr,
        Ttr,
        block_size=cfg["block_size"],
        n_pairs=cfg["train_pairs"],
        pos_fraction=cfg["pos_fraction"],
        include_ba=cfg["include_ba"],
        seed=cfg["seed"],
        within_block_replace=cfg["within_block_replace"],
        max_row_reuse=cfg["max_row_reuse"],
    )
    val_ds = BlockPairDataset(
        Xva,
        Tva,
        block_size=cfg["block_size"],
        n_pairs=cfg["val_pairs"],
        pos_fraction=cfg["pos_fraction"],
        include_ba=cfg["include_ba"],
        seed=cfg["seed"] + 1,
        within_block_replace=cfg["within_block_replace"],
        max_row_reuse=None,
    )
    test_ds = BlockPairDataset(
        Xte,
        Tte,
        block_size=cfg["block_size"],
        n_pairs=cfg["test_pairs"],
        pos_fraction=cfg["pos_fraction"],
        include_ba=cfg["include_ba"],
        seed=cfg["seed"] + 2,
        within_block_replace=cfg["within_block_replace"],
        max_row_reuse=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    model = SiameseBlockPairClassifier(
        input_dim=input_dim,
        emb_dim=cfg.get("emb_dim", 16),
        c1=cfg.get("c1", 16),
        c2=cfg.get("c2", 32),
        kernel_size=cfg.get("kernel_size", 3),
        encoder_dropout=cfg.get("encoder_dropout", 0.2),
        pooling=cfg.get("pooling", "mean"),
        head_hidden=cfg.get("head_hidden", 32),
        head_dropout=cfg.get("head_dropout", 0.5),
    )

    model, history = train_block_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
    )

    plot_history(history)

    test_scores, test_labels, test_loss = predict_scores_and_loss(model, test_loader, device)
    test_auc, test_acc, test_fpr, test_tpr = compute_metrics(test_scores, test_labels)

    print(f"\n=== TEST RESULTS (Block CNN, block_size = {cfg['block_size']}) ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test AUC : {test_auc:.4f}")
    print(f"Test ACC : {test_acc:.4f}")
    print(
        "Score summary:",
        float(test_scores.min()),
        float(test_scores.mean()),
        float(test_scores.max()),
    )

    plot_roc_curve(
        test_fpr,
        test_tpr,
        test_auc,
        title=f"ROC Curve (Test) — Block CNN, block_size={cfg['block_size']}",
    )

    return model, history


# Example usage:
# model, history = run_experiment(preprocess, X_train, X_val, X_test, T_train, T_val, T_test)