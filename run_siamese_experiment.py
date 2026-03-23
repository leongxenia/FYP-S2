"""
Master runner for the retained Siamese experiments.

Available experiment names:
    - "row_cnn"
    - "block_cnn"
"""

from exp_01_row_cnn import run_experiment as run_row_cnn
from exp_02_block_cnn import run_experiment as run_block_cnn


EXPERIMENT_REGISTRY = {
    "row_cnn": {
        "runner": run_row_cnn,
        "title": "Experiment 1: Row CNN Siamese Pair Classifier",
        "description": (
            "Row-level Siamese pair classifier using a shared 1D CNN encoder "
            "and same/different pair labels."
        ),
    },
    "block_cnn": {
        "runner": run_block_cnn,
        "title": "Experiment 2: Block-Based CNN Siamese Model",
        "description": (
            "Block-level Siamese classifier using grouped rows to reduce row-level "
            "instability while keeping the pair-classification framework."
        ),
    },
}


def print_available_experiments():
    print("Available experiment names:\n")
    for key, meta in EXPERIMENT_REGISTRY.items():
        print(f"- {key}")
        print(f"  {meta['title']}")
        print(f"  {meta['description']}\n")


def get_experiment_runner(experiment_name: str):
    if experiment_name not in EXPERIMENT_REGISTRY:
        valid = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown experiment_name='{experiment_name}'. "
            f"Valid options are: {valid}"
        )
    return EXPERIMENT_REGISTRY[experiment_name]["runner"]


def run_experiment_by_name(
    experiment_name,
    preprocess,
    X_train,
    X_val,
    X_test,
    T_train,
    T_val,
    T_test,
):
    if experiment_name not in EXPERIMENT_REGISTRY:
        valid = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown experiment_name='{experiment_name}'. "
            f"Valid options are: {valid}"
        )

    meta = EXPERIMENT_REGISTRY[experiment_name]
    print("=" * 80)
    print(meta["title"])
    print(meta["description"])
    print("=" * 80)

    runner = meta["runner"]
    model, history = runner(
        preprocess=preprocess,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        T_train=T_train,
        T_val=T_val,
        T_test=T_test,
    )
    return model, history


def run_all_experiments(
    preprocess_factory,
    X_train,
    X_val,
    X_test,
    T_train,
    T_val,
    T_test,
):
    results = {}

    for experiment_name in EXPERIMENT_REGISTRY.keys():
        print("\n" + "#" * 100)
        print(f"Running: {experiment_name}")
        print("#" * 100)

        preprocess = preprocess_factory()

        model, history = run_experiment_by_name(
            experiment_name=experiment_name,
            preprocess=preprocess,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            T_train=T_train,
            T_val=T_val,
            T_test=T_test,
        )

        results[experiment_name] = {
            "model": model,
            "history": history,
        }

    return results