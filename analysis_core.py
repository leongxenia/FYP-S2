from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import copy
import io
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Shared constants and helpers
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["Device", "Location"]
NUMERICAL_FEATURES = ["Page Views", "Time Spent"]
DROP_COLS = ["Group", "Conversion", "User ID"]
KEEP_COLS_BASE = ["Device", "Location", "Page Views", "Time Spent"]
CANDIDATE_TOPK = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90, 0.80, 1.00]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def preprocess_factory() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", numeric_pipe, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )


def _map_treatment_and_outcome(df: pd.DataFrame):
    T = (
        df["Group"]
        .astype(str).str.strip().str.upper()
        .map({"A": 0, "B": 1})
        .astype(int)
    )
    y = (
        df["Conversion"]
        .astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
        .astype(int)
    )
    return T, y


def prepare_project_data(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    T, y = _map_treatment_and_outcome(df)
    X = df.drop(columns=DROP_COLS, errors="ignore").copy()

    strat = T.astype(str) + "_" + y.astype(str)
    X_temp, X_test, T_temp, T_test, y_temp, y_test = train_test_split(
        X,
        T,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=strat,
    )

    strat_temp = T_temp.astype(str) + "_" + y_temp.astype(str)
    X_train, X_val, T_train, T_val, y_train, y_val = train_test_split(
        X_temp,
        T_temp,
        y_temp,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=strat_temp,
    )

    rate_A_train = float(y_train[T_train == 0].mean())
    rate_B_train = float(y_train[T_train == 1].mean())
    ate_train = rate_B_train - rate_A_train

    rate_A_test = float(y_test[T_test == 0].mean())
    rate_B_test = float(y_test[T_test == 1].mean())
    ate_test = rate_B_test - rate_A_test

    return {
        "df": df,
        "X": X,
        "T": T,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "T_train": T_train,
        "T_val": T_val,
        "T_test": T_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "preprocess_factory": preprocess_factory,
        "prop_val": float(np.mean(T_val)),
        "prop_test": float(np.mean(T_test)),
        "rate_A_train": rate_A_train,
        "rate_B_train": rate_B_train,
        "ate_train": ate_train,
        "rate_A_test": rate_A_test,
        "rate_B_test": rate_B_test,
        "ate_test": ate_test,
    }


# -----------------------------------------------------------------------------
# Uplift modelling helpers (notebook-aligned)
# -----------------------------------------------------------------------------

def predict_uplift_tlearner(model_A, model_B, X_input):
    pA = model_A.predict_proba(X_input)[:, 1]
    pB = model_B.predict_proba(X_input)[:, 1]
    uplift = pB - pA
    return uplift, pA, pB


def make_topk_policy(uplift_scores: np.ndarray, topk: float) -> np.ndarray:
    n = len(uplift_scores)
    k = int(np.ceil(n * topk))
    order = np.argsort(-uplift_scores)
    policy = np.zeros(n, dtype=int)
    policy[order[:k]] = 1
    return policy


def ips_policy_value(y_true: np.ndarray, T_logged: np.ndarray, policy_T: np.ndarray, prop: float) -> float:
    p = np.where(policy_T == 1, prop, 1.0 - prop)
    w = (T_logged == policy_T).astype(float) / np.clip(p, 1e-6, 1.0)
    return float(np.mean(y_true * w))


def qini_curve(uplift_scores: np.ndarray, T_logged: np.ndarray, y_true: np.ndarray):
    dfq = pd.DataFrame({"uplift": uplift_scores, "T": T_logged, "y": y_true}).sort_values("uplift", ascending=False)

    treat = (dfq["T"].values == 1).astype(int)
    ctrl = (dfq["T"].values == 0).astype(int)
    yv = dfq["y"].values

    cum_treat = np.cumsum(treat)
    cum_ctrl = np.cumsum(ctrl)
    cum_y_treat = np.cumsum(yv * treat)
    cum_y_ctrl = np.cumsum(yv * ctrl)

    ratio = np.divide(cum_treat, np.clip(cum_ctrl, 1, None))
    q = cum_y_treat - ratio * cum_y_ctrl

    frac = np.arange(1, len(dfq) + 1) / len(dfq)
    return frac, q


def random_qini_baseline(T_logged: np.ndarray, y_true: np.ndarray, n_rep: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    T_logged = np.asarray(T_logged)
    y_true = np.asarray(y_true)
    n = len(y_true)

    qs = []
    for _ in range(n_rep):
        random_scores = rng.normal(size=n)
        frac, q = qini_curve(random_scores, T_logged, y_true)
        qs.append(q)

    q_mean = np.mean(np.vstack(qs), axis=0)
    frac = np.arange(1, n + 1) / n
    return frac, q_mean


def auqc(uplift_scores: np.ndarray, T_logged: np.ndarray, y_true: np.ndarray) -> float:
    frac, q = qini_curve(uplift_scores, T_logged, y_true)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(q, frac))


def bootstrap_uplift_summaries(uplift_by_model: dict, B: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    any_model = next(iter(uplift_by_model))
    n = len(uplift_by_model[any_model])

    rows = []
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        for model, u in uplift_by_model.items():
            u = np.asarray(u, dtype=float)
            ub = u[idx]
            rows.append({
                "bootstrap": b,
                "model": model,
                "mean_uplift": float(np.mean(ub)),
                "median_uplift": float(np.median(ub)),
                "share_pos": float(np.mean(ub > 0)),
            })
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return mean, lo, hi


def fmt_ci(lo: float, hi: float, digits: int = 4) -> str:
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def bootstrap_ate_ci(y_arr: np.ndarray, T_arr: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_treat = np.where(T_arr == 1)[0]
    idx_ctrl = np.where(T_arr == 0)[0]
    ates = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        samp_t = rng.choice(idx_treat, size=len(idx_treat), replace=True)
        samp_c = rng.choice(idx_ctrl, size=len(idx_ctrl), replace=True)
        rate_t = float(np.mean(y_arr[samp_t]))
        rate_c = float(np.mean(y_arr[samp_c]))
        ates[b] = rate_t - rate_c

    return ates


def bootstrap_policy_metrics_stratified(
    uplift_scores: np.ndarray,
    y_arr: np.ndarray,
    T_arr: np.ndarray,
    topk: float,
    prop: float,
    n_boot: int = 2000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    uplift_scores = np.asarray(uplift_scores)
    y_arr = np.asarray(y_arr)
    T_arr = np.asarray(T_arr)

    idx_treat = np.where(T_arr == 1)[0]
    idx_ctrl = np.where(T_arr == 0)[0]

    n_t = len(idx_treat)
    n_c = len(idx_ctrl)

    ips_vals = np.empty(n_boot, dtype=float)
    baseA_vals = np.empty(n_boot, dtype=float)
    auqc_vals = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        samp_t = rng.choice(idx_treat, size=n_t, replace=True)
        samp_c = rng.choice(idx_ctrl, size=n_c, replace=True)
        idx = np.concatenate([samp_t, samp_c])

        y_b = y_arr[idx]
        T_b = T_arr[idx]
        u_b = uplift_scores[idx]

        policy_b = make_topk_policy(u_b, topk=topk)
        ips_vals[b] = ips_policy_value(y_b, T_b, policy_b, prop=prop)
        baseA_vals[b] = float(np.mean(y_b[T_b == 0]))
        auqc_vals[b] = auqc(u_b, T_b, y_b)

    lift_vs_A = ips_vals - baseA_vals
    return {
        "ips_vals": ips_vals,
        "baseA_vals": baseA_vals,
        "lift_vs_A": lift_vs_A,
        "auqc_vals": auqc_vals,
    }


def run_t_learner_analysis(
    bundle: dict,
    manual_topk: float = 0.80,
    n_boot: int = 2000,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    if progress_callback is None:
        progress_callback = lambda _msg: None

    base_learners = {
        "LogReg": LogisticRegression(max_iter=2000, random_state=42),
        "HistGB": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=42
        ),
        "RandForest": RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(16,),
            alpha=1e-3,
            learning_rate_init=3e-4,
            max_iter=2000,
            early_stopping=False,
            random_state=42,
        ),
        "NaiveBayes": GaussianNB(),
    }

    X_train = bundle["X_train"]
    X_val = bundle["X_val"]
    X_test = bundle["X_test"]
    T_train = bundle["T_train"]
    T_val = np.asarray(bundle["T_val"])
    T_test = np.asarray(bundle["T_test"])
    y_val = np.asarray(bundle["y_val"])
    y_test = np.asarray(bundle["y_test"])
    prop_val = float(bundle["prop_val"])
    prop_test = float(bundle["prop_test"])

    models_A = {}
    models_B = {}
    uplift_val_by_model = {}
    uplift_test_by_model = {}
    auqc_by_model = {}
    qini_lines = {}
    val_topk_tables = {}
    pred_means = {}

    baseline_A = float(np.mean(y_test[T_test == 0]))
    baseline_B = float(np.mean(y_test[T_test == 1]))

    for i, (name, clf) in enumerate(base_learners.items(), start=1):
        progress_callback(f"Training uplift model {i}/{len(base_learners)}: {name}")

        model_A = Pipeline([("prep", clone(preprocess_factory())), ("clf", clone(clf))])
        model_B = Pipeline([("prep", clone(preprocess_factory())), ("clf", clone(clf))])

        model_A.fit(X_train[T_train == 0], bundle["y_train"][T_train == 0])
        model_B.fit(X_train[T_train == 1], bundle["y_train"][T_train == 1])

        models_A[name] = model_A
        models_B[name] = model_B

        uplift_val, pA_val, pB_val = predict_uplift_tlearner(model_A, model_B, X_val)
        uplift_test, pA_test, pB_test = predict_uplift_tlearner(model_A, model_B, X_test)

        uplift_val_by_model[name] = uplift_val
        uplift_test_by_model[name] = uplift_test

        auqc_test = auqc(uplift_test, T_test, y_test)
        auqc_by_model[name] = auqc_test

        frac, q = qini_curve(uplift_test, T_test, y_test)
        qini_lines[name] = {"fraction": frac, "qini": q, "auqc": auqc_test}

        pred_means[name] = {
            "mean_pA_test": float(np.mean(pA_test)),
            "mean_pB_test": float(np.mean(pB_test)),
            "mean_uplift_test": float(np.mean(uplift_test)),
            "std_uplift_test": float(np.std(uplift_test)),
        }

        val_rows = []
        for k in CANDIDATE_TOPK:
            policy_val = make_topk_policy(uplift_val, topk=k)
            v_ips_val = ips_policy_value(y_val, T_val, policy_val, prop=prop_val)
            val_rows.append({"topk": k, "IPS_value_val": v_ips_val})
        val_topk_tables[name] = pd.DataFrame(val_rows).sort_values("IPS_value_val", ascending=False)

    progress_callback("Summarising uplift policies and bootstrap intervals")

    model_summary = []
    for name, uplift_test in uplift_test_by_model.items():
        policy_test = make_topk_policy(uplift_test, topk=manual_topk)
        v_ips_test = ips_policy_value(y_test, T_test, policy_test, prop=prop_test)
        model_summary.append({
            "model": name,
            "topk_used": manual_topk,
            "IPS_test_policy": float(v_ips_test),
            "baseline_A_test": baseline_A,
            "baseline_B_test": baseline_B,
            "AUQC_test": float(auqc_by_model[name]),
        })
    summary_df = pd.DataFrame(model_summary).sort_values(["IPS_test_policy", "AUQC_test"], ascending=False).reset_index(drop=True)

    # Keep mean-uplift bootstrap at notebook setting B=100.
    boot_df = bootstrap_uplift_summaries(uplift_test_by_model, B=100, seed=42)

    ates_boot = bootstrap_ate_ci(y_test, T_test, n_boot=n_boot, seed=42)
    ate_mean, ate_lo, ate_hi = bootstrap_ci(ates_boot, alpha=0.05)

    bootstrap_rows = []
    bootstrap_metrics_by_model = {}
    mean_uplift_crosscheck = []
    for name, uplift_scores in uplift_test_by_model.items():
        metrics = bootstrap_policy_metrics_stratified(
            uplift_scores=uplift_scores,
            y_arr=y_test,
            T_arr=T_test,
            topk=float(manual_topk),
            prop=prop_test,
            n_boot=n_boot,
            seed=42,
        )
        bootstrap_metrics_by_model[name] = metrics

        ips_mean, ips_lo, ips_hi = bootstrap_ci(metrics["ips_vals"])
        liftA_mean, liftA_lo, liftA_hi = bootstrap_ci(metrics["lift_vs_A"])
        auqc_mean, auqc_lo, auqc_hi = bootstrap_ci(metrics["auqc_vals"])

        bootstrap_rows.append({
            "model": name,
            "topk_used": float(manual_topk),
            "IPS_mean": ips_mean,
            "IPS_CI": fmt_ci(ips_lo, ips_hi),
            "Lift_vs_A_mean": liftA_mean,
            "Lift_vs_A_CI": fmt_ci(liftA_lo, liftA_hi),
            "AUQC_mean": auqc_mean,
            "AUQC_CI": fmt_ci(auqc_lo, auqc_hi),
        })

        boot_mean = float(boot_df.loc[boot_df["model"] == name, "mean_uplift"].mean())
        direct_mean = float(pred_means[name]["mean_uplift_test"])
        mean_uplift_crosscheck.append({
            "model": name,
            "direct_mean_uplift": direct_mean,
            "bootstrap_mean_uplift": boot_mean,
            "difference": boot_mean - direct_mean,
        })

    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(["IPS_mean", "AUQC_mean"], ascending=False).reset_index(drop=True)
    mean_uplift_crosscheck_df = pd.DataFrame(mean_uplift_crosscheck).sort_values("model").reset_index(drop=True)

    frac_r, q_r = random_qini_baseline(T_test, y_test, n_rep=200, seed=42)

    return {
        "models_A": models_A,
        "models_B": models_B,
        "uplift_val_by_model": uplift_val_by_model,
        "uplift_test_by_model": uplift_test_by_model,
        "summary_df": summary_df,
        "boot_df": boot_df,
        "bootstrap_df": bootstrap_df,
        "bootstrap_metrics_by_model": bootstrap_metrics_by_model,
        "ates_boot": ates_boot,
        "ate_bootstrap_summary": {
            "mean": ate_mean,
            "lo": ate_lo,
            "hi": ate_hi,
            "formatted": fmt_ci(ate_lo, ate_hi),
        },
        "manual_topk": manual_topk,
        "pred_means": pred_means,
        "mean_uplift_crosscheck": mean_uplift_crosscheck_df,
        "val_topk_tables": val_topk_tables,
        "qini_lines": qini_lines,
        "qini_random": {"fraction": frac_r, "qini": q_r},
        "auqc_by_model": auqc_by_model,
    }


# -----------------------------------------------------------------------------
# Exploratory HTE analysis (notebook-aligned)
# -----------------------------------------------------------------------------

def build_hte_df(df_raw: pd.DataFrame, T_series: pd.Series, y_series: pd.Series) -> pd.DataFrame:
    out = df_raw.copy()
    out["T"] = T_series.values
    out["y"] = y_series.values
    keep_cols = ["T", "y"] + KEEP_COLS_BASE
    out = out[keep_cols].dropna().copy()
    out["T"] = out["T"].astype(int)
    out["y"] = out["y"].astype(int)
    return out


def segment_uplift_table(df_in: pd.DataFrame, seg_col: str, decimals: int = 4) -> pd.DataFrame:
    rows = []
    for seg_value, x in df_in.groupby(seg_col, dropna=False):
        x_control = x[x["T"] == 0]
        x_treatment = x[x["T"] == 1]

        n_control = len(x_control)
        n_treatment = len(x_treatment)
        n_total = len(x)

        control_rate = x_control["y"].mean() if n_control > 0 else np.nan
        treatment_rate = x_treatment["y"].mean() if n_treatment > 0 else np.nan
        uplift = treatment_rate - control_rate if pd.notna(control_rate) and pd.notna(treatment_rate) else np.nan

        rows.append({
            seg_col: seg_value,
            "n_total": n_total,
            "n_control": n_control,
            "n_treatment": n_treatment,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "uplift": uplift,
        })

    out = pd.DataFrame(rows)
    rate_cols = ["control_rate", "treatment_rate", "uplift"]
    out[rate_cols] = out[rate_cols].round(decimals)
    return out.sort_values(["uplift", "n_total"], ascending=[False, False]).reset_index(drop=True)


def make_quantile_bins(series: pd.Series, q=4):
    clean_series = series.dropna()
    try:
        temp_binned = pd.qcut(clean_series, q=q, duplicates="drop")
        n_bins = temp_binned.cat.categories.size
        labels = [f"Q{i+1}" for i in range(n_bins)]
        binned = pd.qcut(clean_series, q=q, labels=labels, duplicates="drop")
        out = pd.Series(index=series.index, dtype="object")
        out.loc[clean_series.index] = binned
        out = pd.Series(pd.Categorical(out, categories=labels, ordered=True), index=series.index, name=series.name)
        return out, n_bins
    except ValueError:
        empty = pd.Series(index=series.index, dtype="object", name=series.name)
        return empty, 0


def run_hte_analysis(bundle: dict) -> dict:
    df_hte = build_hte_df(bundle["df"], bundle["T"], bundle["y"])

    formula = '''
    y ~ T
      + Q("Page Views") + Q("Time Spent")
      + C(Device) + C(Location)
      + T:Q("Page Views") + T:Q("Time Spent")
      + T:C(Device) + T:C(Location)
    '''

    glm_int = smf.glm(
        formula=formula,
        data=df_hte,
        family=sm.families.Binomial(),
    ).fit()

    params = glm_int.params
    pvals = glm_int.pvalues
    interaction_rows = []
    for term in params.index:
        if "T:" in term:
            interaction_rows.append({
                "term": term,
                "coef_logodds": float(params[term]),
                "odds_ratio": float(np.exp(params[term])),
                "p_value": float(pvals[term]),
            })
    interaction_df = pd.DataFrame(interaction_rows).sort_values("p_value").reset_index(drop=True)

    device_uplift = segment_uplift_table(df_hte, "Device")
    location_uplift = segment_uplift_table(df_hte, "Location")

    df_hte = df_hte.copy()
    df_hte["PageViews_bin"], n_pv_bins = make_quantile_bins(df_hte["Page Views"], q=4)
    df_hte["TimeSpent_bin"], n_ts_bins = make_quantile_bins(df_hte["Time Spent"], q=4)

    if n_pv_bins >= 2:
        pv_uplift = segment_uplift_table(df_hte.dropna(subset=["PageViews_bin"]), "PageViews_bin")
        pv_uplift = pv_uplift.sort_values("PageViews_bin").reset_index(drop=True)
    else:
        pv_uplift = None

    if n_ts_bins >= 2:
        ts_uplift = segment_uplift_table(df_hte.dropna(subset=["TimeSpent_bin"]), "TimeSpent_bin")
        ts_uplift = ts_uplift.sort_values("TimeSpent_bin").reset_index(drop=True)
    else:
        ts_uplift = None

    buf = io.StringIO()
    buf.write(glm_int.summary().as_text())

    return {
        "df_hte": df_hte,
        "glm_model": glm_int,
        "glm_summary_text": buf.getvalue(),
        "interaction_df": interaction_df,
        "device_uplift": device_uplift,
        "location_uplift": location_uplift,
        "pv_uplift": pv_uplift,
        "ts_uplift": ts_uplift,
    }


# -----------------------------------------------------------------------------
# Siamese wrappers
# -----------------------------------------------------------------------------

def _ensure_siamese_imports():
    # Support either a complete app folder or the original uploaded module layout.
    search_paths = [
        Path(__file__).resolve().parent,
        Path.cwd(),
        Path('/mnt/data'),
    ]
    for p in search_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Handle user uploads that had " (1)" suffixes by creating aliases when possible.
    base = Path('/mnt/data')
    src1 = base / 'exp_01_baseline_cnn_pair (1).py'
    dst1 = base / 'exp_01_baseline_cnn_pair.py'
    if src1.exists() and not dst1.exists():
        dst1.write_text(src1.read_text())

    src2 = base / 'siamese_utils (1).py'
    dst2 = base / 'siamese_utils.py'
    if src2.exists() and not dst2.exists():
        dst2.write_text(src2.read_text())

    from run_siamese_experiment import run_experiment_by_name
    from siamese_config import BASE_PAIR_CONFIG, BAG_CONFIG
    from siamese_data import prepare_encoded_data, PairDataset, BagPairDataset
    from siamese_eval import predict_pair_scores, evaluate_pairs, predict_scores_and_loss, compute_metrics
    from siamese_models import SiameseCNNPairClassifier, SiameseBagPairClassifier
    from siamese_utils import set_seed, get_device
    import torch
    from torch.utils.data import DataLoader

    return {
        "run_experiment_by_name": run_experiment_by_name,
        "BASE_PAIR_CONFIG": BASE_PAIR_CONFIG,
        "BAG_CONFIG": BAG_CONFIG,
        "prepare_encoded_data": prepare_encoded_data,
        "PairDataset": PairDataset,
        "BagPairDataset": BagPairDataset,
        "predict_pair_scores": predict_pair_scores,
        "evaluate_pairs": evaluate_pairs,
        "predict_scores_and_loss": predict_scores_and_loss,
        "compute_metrics": compute_metrics,
        "SiameseCNNPairClassifier": SiameseCNNPairClassifier,
        "SiameseBagPairClassifier": SiameseBagPairClassifier,
        "set_seed": set_seed,
        "get_device": get_device,
        "torch": torch,
        "DataLoader": DataLoader,
    }


def _preprocess_for_siamese(bundle: dict):
    return preprocess_factory()


def _downscale_cfg(cfg: dict, mode: str) -> dict:
    cfg2 = copy.deepcopy(cfg)
    if mode == 'baseline':
        cfg2.update({
            'train_pairs': min(cfg2.get('train_pairs', 200000), 40000),
            'val_pairs': min(cfg2.get('val_pairs', 60000), 12000),
            'test_pairs': min(cfg2.get('test_pairs', 80000), 16000),
            'max_epochs': min(cfg2.get('max_epochs', 30), 12),
            'patience': min(cfg2.get('patience', 6), 4),
            'batch_size': min(cfg2.get('batch_size', 512), 256),
        })
    else:
        cfg2.update({
            'train_pairs': min(cfg2.get('train_pairs', 30000), 12000),
            'val_pairs': min(cfg2.get('val_pairs', 20000), 8000),
            'test_pairs': min(cfg2.get('test_pairs', 30000), 12000),
            'max_epochs': min(cfg2.get('max_epochs', 30), 12),
            'patience': min(cfg2.get('patience', 6), 4),
            'batch_size': min(cfg2.get('batch_size', 256), 256),
        })
    return cfg2


def _run_baseline_siamese_impl(bundle: dict, fast_mode: bool):
    m = _ensure_siamese_imports()
    cfg = copy.deepcopy(m['BASE_PAIR_CONFIG'])
    if fast_mode:
        cfg = _downscale_cfg(cfg, 'baseline')

    m['set_seed'](cfg['seed'])
    device = m['get_device']()

    preprocess = _preprocess_for_siamese(bundle)
    Xtr, Xva, Xte, Ttr, Tva, Tte = m['prepare_encoded_data'](
        preprocess,
        bundle['X_train'], bundle['X_val'], bundle['X_test'],
        bundle['T_train'], bundle['T_val'], bundle['T_test'],
    )

    train_ds = m['PairDataset'](Xtr, Ttr, n_pairs=cfg['train_pairs'], pos_fraction=cfg['pos_fraction'], seed=cfg['seed'], include_ba=cfg['include_ba'])
    val_ds = m['PairDataset'](Xva, Tva, n_pairs=cfg['val_pairs'], pos_fraction=cfg['pos_fraction'], seed=cfg['seed'] + 1, include_ba=cfg['include_ba'])
    test_ds = m['PairDataset'](Xte, Tte, n_pairs=cfg['test_pairs'], pos_fraction=cfg['pos_fraction'], seed=cfg['seed'] + 2, include_ba=cfg['include_ba'])

    train_loader = m['DataLoader'](train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = m['DataLoader'](val_ds, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = m['DataLoader'](test_ds, batch_size=cfg['batch_size'], shuffle=False)
    train_eval_loader = m['DataLoader'](train_ds, batch_size=cfg['batch_size'], shuffle=False, drop_last=False)

    model = m['SiameseCNNPairClassifier'](input_len=Xtr.shape[1], emb_dim=32, hidden=64)
    from siamese_trainers import train_pair_classifier
    model, history = train_pair_classifier(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg['lr'],
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
    )

    val_scores, val_labels = m['predict_pair_scores'](model, val_loader, device)
    val_auc, val_acc, val_fpr, val_tpr = m['evaluate_pairs'](val_scores, val_labels)

    test_scores, test_labels = m['predict_pair_scores'](model, test_loader, device)
    test_auc, test_acc, test_fpr, test_tpr = m['evaluate_pairs'](test_scores, test_labels)

    return {
        'title': 'Experiment 1: Baseline CNN Siamese Pair Classifier',
        'history': history,
        'device': str(device),
        'encoded_shapes': {'train': list(Xtr.shape), 'val': list(Xva.shape), 'test': list(Xte.shape)},
        'config': cfg,
        'val': {
            'auc': float(val_auc),
            'acc': float(val_acc),
            'fpr': val_fpr.tolist(),
            'tpr': val_tpr.tolist(),
            'score_summary': {
                'min': float(np.min(val_scores)),
                'mean': float(np.mean(val_scores)),
                'max': float(np.max(val_scores)),
            },
        },
        'test': {
            'auc': float(test_auc),
            'acc': float(test_acc),
            'fpr': test_fpr.tolist(),
            'tpr': test_tpr.tolist(),
            'score_summary': {
                'min': float(np.min(test_scores)),
                'mean': float(np.mean(test_scores)),
                'max': float(np.max(test_scores)),
            },
        },
    }


def _run_bag_siamese_impl(bundle: dict, fast_mode: bool):
    m = _ensure_siamese_imports()
    cfg = copy.deepcopy(m['BAG_CONFIG'])
    if fast_mode:
        cfg = _downscale_cfg(cfg, 'bag')

    m['set_seed'](cfg['seed'])
    device = m['get_device']()

    preprocess = _preprocess_for_siamese(bundle)
    Xtr, Xva, Xte, Ttr, Tva, Tte = m['prepare_encoded_data'](
        preprocess,
        bundle['X_train'], bundle['X_val'], bundle['X_test'],
        bundle['T_train'], bundle['T_val'], bundle['T_test'],
    )

    train_ds = m['BagPairDataset'](
        Xtr, Ttr,
        bag_size=cfg['bag_size'], n_pairs=cfg['train_pairs'], pos_fraction=cfg['pos_fraction'],
        include_ba=cfg['include_ba'], seed=cfg['seed'], within_bag_replace=cfg['within_bag_replace'], max_row_reuse=cfg['max_row_reuse'],
    )
    val_ds = m['BagPairDataset'](
        Xva, Tva,
        bag_size=cfg['bag_size'], n_pairs=cfg['val_pairs'], pos_fraction=cfg['pos_fraction'],
        include_ba=cfg['include_ba'], seed=cfg['seed'] + 1, within_bag_replace=cfg['within_bag_replace'], max_row_reuse=None,
    )
    test_ds = m['BagPairDataset'](
        Xte, Tte,
        bag_size=cfg['bag_size'], n_pairs=cfg['test_pairs'], pos_fraction=cfg['pos_fraction'],
        include_ba=cfg['include_ba'], seed=cfg['seed'] + 2, within_bag_replace=cfg['within_bag_replace'], max_row_reuse=None,
    )

    train_loader = m['DataLoader'](train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = m['DataLoader'](val_ds, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = m['DataLoader'](test_ds, batch_size=cfg['batch_size'], shuffle=False)

    model = m['SiameseBagPairClassifier'](
        input_dim=Xtr.shape[1],
        emb_dim=cfg.get('emb_dim', 16),
        c1=cfg.get('c1', 16),
        c2=cfg.get('c2', 32),
        kernel_size=cfg.get('kernel_size', 3),
        encoder_dropout=cfg.get('encoder_dropout', 0.2),
        pooling=cfg.get('pooling', 'mean'),
        head_hidden=cfg.get('head_hidden', 32),
        head_dropout=cfg.get('head_dropout', 0.5),
    )
    from siamese_trainers import train_bag_classifier
    model, history = train_bag_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
    )

    test_scores, test_labels, test_loss = m['predict_scores_and_loss'](model, test_loader, device)
    test_auc, test_acc, test_fpr, test_tpr = m['compute_metrics'](test_scores, test_labels)

    # Validation metrics for app plotting convenience.
    val_scores, val_labels, val_loss = m['predict_scores_and_loss'](model, val_loader, device)
    val_auc, val_acc, val_fpr, val_tpr = m['compute_metrics'](val_scores, val_labels)

    return {
        'title': 'Experiment 2: Bag-Based CNN Siamese Model',
        'history': history,
        'device': str(device),
        'encoded_shapes': {'train': list(Xtr.shape), 'val': list(Xva.shape), 'test': list(Xte.shape)},
        'config': cfg,
        'val': {
            'auc': float(val_auc),
            'acc': float(val_acc),
            'loss': float(val_loss),
            'fpr': val_fpr.tolist(),
            'tpr': val_tpr.tolist(),
            'score_summary': {
                'min': float(np.min(val_scores)),
                'mean': float(np.mean(val_scores)),
                'max': float(np.max(val_scores)),
            },
        },
        'test': {
            'auc': float(test_auc),
            'acc': float(test_acc),
            'loss': float(test_loss),
            'fpr': test_fpr.tolist(),
            'tpr': test_tpr.tolist(),
            'score_summary': {
                'min': float(np.min(test_scores)),
                'mean': float(np.mean(test_scores)),
                'max': float(np.max(test_scores)),
            },
        },
    }


def run_baseline_siamese(bundle: dict, fast_mode: bool = True) -> dict:
    return _run_baseline_siamese_impl(bundle, fast_mode=fast_mode)


def run_bag_siamese(bundle: dict, fast_mode: bool = True) -> dict:
    return _run_bag_siamese_impl(bundle, fast_mode=fast_mode)


def summarise_siamese_results(siamese_results: dict) -> pd.DataFrame:
    rows = []
    for key, res in siamese_results.items():
        rows.append({
            'experiment': key,
            'title': res['title'],
            'test_auc': float(res['test']['auc']),
            'test_acc': float(res['test']['acc']),
            'best_val_auc': float(max(res['history'].get('val_auc', [np.nan]))),
            'epochs_run': int(len(res['history'].get('epoch', []))),
        })
    return pd.DataFrame(rows).sort_values(['test_auc', 'best_val_auc'], ascending=False).reset_index(drop=True)
