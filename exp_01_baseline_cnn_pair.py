import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from siamese_config import BASE_PAIR_CONFIG
from siamese_utils import set_seed, get_device
from siamese_data import prepare_encoded_data, PairDataset
from siamese_models import SiameseCNNPairClassifier
from siamese_trainers import train_pair_classifier
from siamese_eval import predict_pair_scores, evaluate_pairs, plot_roc_curve



def plot_loss_curves(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Experiment 1: Epoch vs Loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()



def plot_metric_curves(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_auc"], label="train_auc")
    plt.plot(history["epoch"], history["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Experiment 1: Epoch vs AUC")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_acc"], label="train_acc")
    plt.plot(history["epoch"], history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Experiment 1: Epoch vs Accuracy")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()



def run_experiment(preprocess, X_train, X_val, X_test, T_train, T_val, T_test):
    cfg = BASE_PAIR_CONFIG
    set_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    Xtr, Xva, Xte, Ttr, Tva, Tte = prepare_encoded_data(
        preprocess, X_train, X_val, X_test, T_train, T_val, T_test
    )
    print("Encoded shapes:", Xtr.shape, Xva.shape, Xte.shape)

    input_len = Xtr.shape[1]

    train_ds = PairDataset(
        Xtr,
        Ttr,
        n_pairs=cfg["train_pairs"],
        pos_fraction=cfg["pos_fraction"],
        seed=cfg["seed"],
        include_ba=cfg["include_ba"],
    )
    val_ds = PairDataset(
        Xva,
        Tva,
        n_pairs=cfg["val_pairs"],
        pos_fraction=cfg["pos_fraction"],
        seed=cfg["seed"] + 1,
        include_ba=cfg["include_ba"],
    )
    test_ds = PairDataset(
        Xte,
        Tte,
        n_pairs=cfg["test_pairs"],
        pos_fraction=cfg["pos_fraction"],
        seed=cfg["seed"] + 2,
        include_ba=cfg["include_ba"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)

    model = SiameseCNNPairClassifier(input_len=input_len, emb_dim=32, hidden=64)

    model, history = train_pair_classifier(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg["lr"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
    )

    plot_loss_curves(history)
    plot_metric_curves(history)

    val_scores, val_labels = predict_pair_scores(model, val_loader, device)
    val_auc, val_acc, val_fpr, val_tpr = evaluate_pairs(val_scores, val_labels)
    print(f"\nVAL: AUC={val_auc:.4f} | ACC={val_acc:.4f}")
    plot_roc_curve(val_fpr, val_tpr, val_auc, title="Experiment 1 ROC Curve (Validation)")

    test_scores, test_labels = predict_pair_scores(model, test_loader, device)
    test_auc, test_acc, test_fpr, test_tpr = evaluate_pairs(test_scores, test_labels)
    print(f"\nTEST: AUC={test_auc:.4f} | ACC={test_acc:.4f}")
    print(
        "Test score summary:",
        float(test_scores.min()),
        float(test_scores.mean()),
        float(test_scores.max()),
    )
    plot_roc_curve(test_fpr, test_tpr, test_auc, title="Experiment 1 ROC Curve (Test)")

    return model, history
