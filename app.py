from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from analysis_core import (
    prepare_project_data,
    run_hte_analysis,
    run_t_learner_analysis,
    run_baseline_siamese,
    run_bag_siamese,
    summarise_siamese_results,
)


sns.set_theme(style="whitegrid", context="notebook")

st.set_page_config(
    page_title="FYP-S2 | ML-Driven A/B Testing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_custom_css(theme_mode: str = "Light", font_scale: float = 1.30) -> str:
    if theme_mode == "Dark":
        colors = {
            "bg": "#0b1220",
            "surface": "#111827",
            "panel": "#0f172a",
            "text": "#e5eefc",
            "muted": "#94a3b8",
            "border": "rgba(148, 163, 184, 0.22)",
            "accent": "#8b9cff",
            "accent_hover": "#a5b4ff",
            "accent_text": "#ffffff",
            "selected_bg": "#6d78ff",
            "selected_text": "#ffffff",
            "selected_border": "rgba(125, 140, 255, 0.55)",
            "card_grad_1": "rgba(139, 156, 255, 0.16)",
            "card_grad_2": "rgba(34, 197, 94, 0.10)",
            "shadow": "0 14px 34px rgba(2, 6, 23, 0.34)",
            "table_header": "#172033",
        }
    else:
        colors = {
            "bg": "#f4f7fb",
            "surface": "#ffffff",
            "panel": "#edf3fb",
            "text": "#102033",
            "muted": "#526172",
            "border": "rgba(15, 23, 42, 0.10)",
            "accent": "#4f46e5",
            "accent_hover": "#4338ca",
            "accent_text": "#ffffff",
            "selected_bg": "#5b5ff5",
            "selected_text": "#ffffff",
            "selected_border": "rgba(91, 95, 245, 0.32)",
            "card_grad_1": "rgba(79, 70, 229, 0.10)",
            "card_grad_2": "rgba(14, 165, 233, 0.08)",
            "shadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
            "table_header": "#f2f6fc",
        }

    base_px = int(round(16 * font_scale))
    small_px = max(14, int(round(base_px * 0.92)))
    h1_px = int(round(base_px * 2.00))
    h2_px = int(round(base_px * 1.62))
    h3_px = int(round(base_px * 1.34))
    metric_px = int(round(base_px * 1.45))

    return f"""
<style>
:root {{
    --app-bg: {colors["bg"]};
    --surface-bg: {colors["surface"]};
    --panel-bg: {colors["panel"]};
    --text-color: {colors["text"]};
    --muted-color: {colors["muted"]};
    --border-color: {colors["border"]};
    --accent-color: {colors["accent"]};
    --accent-hover: {colors["accent_hover"]};
    --accent-text: {colors["accent_text"]};
    --selected-bg: {colors["selected_bg"]};
    --selected-text: {colors["selected_text"]};
    --selected-border: {colors["selected_border"]};
    --card-grad-1: {colors["card_grad_1"]};
    --card-grad-2: {colors["card_grad_2"]};
    --box-shadow: {colors["shadow"]};
    --table-header: {colors["table_header"]};
    --base-font-size: {base_px}px;
    --small-font-size: {small_px}px;
    --h1-font-size: {h1_px}px;
    --h2-font-size: {h2_px}px;
    --h3-font-size: {h3_px}px;
    --metric-font-size: {metric_px}px;
}}

html, body, [class*="css"] {{
    font-size: var(--base-font-size);
}}

body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {{
    background: var(--app-bg);
    color: var(--text-color);
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}

[data-testid="stSidebar"] {{
    background: var(--panel-bg);
    border-right: 1px solid var(--border-color);
}}

[data-testid="stSidebar"] * {{
    color: var(--text-color);
}}

h1, h2, h3, h4, h5, h6, p, label, li, span, div {{
    color: var(--text-color);
}}

h1 {{ font-size: var(--h1-font-size) !important; }}
h2 {{ font-size: var(--h2-font-size) !important; }}
h3 {{ font-size: var(--h3-font-size) !important; }}

p, li, label, .stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stCaptionContainer"] {{
    font-size: var(--base-font-size) !important;
}}

small, .small-note, .stCaption {{
    font-size: var(--small-font-size) !important;
    color: var(--muted-color) !important;
}}

.hero {{
    padding: 1.45rem 1.45rem 1.05rem 1.45rem;
    border: 1px solid var(--border-color);
    border-radius: 22px;
    background: linear-gradient(135deg, var(--card-grad-1), var(--card-grad-2));
    box-shadow: var(--box-shadow);
    margin-bottom: 1.15rem;
}}

.section-card {{
    padding: 1.1rem 1.1rem 0.55rem 1.1rem;
    border: 1px solid var(--border-color);
    border-radius: 18px;
    background: var(--surface-bg);
    box-shadow: var(--box-shadow);
}}

div[data-testid="stMetric"] {{
    background: var(--surface-bg);
    border: 1px solid var(--border-color);
    padding: 0.9rem 1rem;
    border-radius: 18px;
    box-shadow: var(--box-shadow);
}}

div[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {{
    color: var(--muted-color) !important;
    font-size: calc(var(--base-font-size) * 0.95) !important;
}}

[data-testid="stMetricValue"] {{
    font-size: var(--metric-font-size) !important;
    color: var(--text-color) !important;
}}

.stAlert, [data-testid="stAlert"] {{
    background: var(--surface-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
    border-radius: 16px !important;
}}

.stButton > button,
.stDownloadButton > button,
button[kind="primary"],
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {{
    background: var(--accent-color) !important;
    color: var(--accent-text) !important;
    border: 1px solid transparent !important;
    border-radius: 13px !important;
    padding: 0.55rem 1rem !important;
    font-size: var(--base-font-size) !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.18) !important;
}}

.stButton > button *,
.stDownloadButton > button *,
button[kind="primary"] *,
button[kind="secondary"] *,
[data-testid="baseButton-secondary"] *,
[data-testid="baseButton-primary"] * {{
    color: var(--accent-text) !important;
    fill: var(--accent-text) !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {{
    background: var(--accent-hover) !important;
    color: var(--accent-text) !important;
    border-color: transparent !important;
}}

.stButton > button:hover *,
.stDownloadButton > button:hover *,
button[kind="primary"]:hover *,
button[kind="secondary"]:hover * {{
    color: var(--accent-text) !important;
    fill: var(--accent-text) !important;
}}

.stButton > button:focus,
.stDownloadButton > button:focus,
button:focus {{
    outline: 2px solid var(--accent-hover) !important;
    outline-offset: 1px !important;
}}

.stRadio > div,
.stToggle,
.stSlider,
.stMultiSelect,
.stSelectbox,
.stFileUploader,
.stNumberInput,
.stTextInput,
[data-baseweb="select"] > div,
[data-baseweb="base-input"] > div,
[data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"] {{
    color: var(--text-color) !important;
    font-size: var(--base-font-size) !important;
}}

[data-baseweb="select"] > div,
[data-baseweb="base-input"] > div,
[data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"],
.stTextInput input,
.stNumberInput input,
textarea {{
    background: var(--surface-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
}}

[data-baseweb="select"] svg,
.stSelectbox svg,
.stMultiSelect svg,
.stSlider svg {{
    fill: var(--text-color) !important;
}}

[data-baseweb="tag"] {{
    background: var(--selected-bg) !important;
    border: 1px solid var(--selected-border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}

[data-baseweb="tag"] *,
[data-baseweb="tag"] span,
[data-baseweb="tag"] div,
[data-baseweb="tag"] p {{
    color: var(--selected-text) !important;
    fill: var(--selected-text) !important;
}}

[data-baseweb="tag"] svg,
[data-baseweb="tag"] path {{
    fill: var(--selected-text) !important;
    stroke: var(--selected-text) !important;
}}

[data-baseweb="tag"]:hover {{
    filter: brightness(1.03) !important;
}}

[data-baseweb="tab-list"] {{
    gap: 0.35rem;
}}

[data-baseweb="tab-list"] button {{
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--muted-color) !important;
    font-size: var(--base-font-size) !important;
    padding: 0.5rem 0.85rem !important;
}}

[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background: var(--selected-bg) !important;
    color: var(--selected-text) !important;
    border-color: var(--selected-border) !important;
    box-shadow: 0 8px 18px rgba(79, 70, 229, 0.18) !important;
}}

[data-baseweb="tab-list"] button[aria-selected="true"] *,
[data-baseweb="tab-list"] button[aria-selected="true"] span,
[data-baseweb="tab-list"] button[aria-selected="true"] p,
[data-baseweb="tab-list"] button[aria-selected="true"] div {{
    color: var(--selected-text) !important;
    fill: var(--selected-text) !important;
}}

.streamlit-expanderHeader, details {{
    background: var(--surface-bg) !important;
    color: var(--text-color) !important;
    border-radius: 14px !important;
}}

.streamlit-expanderContent, details > div {{
    background: transparent !important;
    color: var(--text-color) !important;
}}

div[data-testid="stDataFrame"],
div[data-testid="stTable"] {{
    background: var(--surface-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px !important;
    overflow: hidden;
}}

div[data-testid="stDataFrame"] * {{
    font-size: calc(var(--base-font-size) * 0.92) !important;
}}

thead tr th {{
    background: var(--table-header) !important;
    color: var(--text-color) !important;
}}

tbody tr td {{
    background: var(--surface-bg) !important;
    color: var(--text-color) !important;
}}

code, pre {{
    background: rgba(255,255,255,0.03) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}}

hr {{
    border-color: var(--border-color);
}}
</style>
"""


def apply_theme_css(theme_mode: str, font_scale: float):
    st.markdown(build_custom_css(theme_mode=theme_mode, font_scale=font_scale), unsafe_allow_html=True)


def apply_matplotlib_theme(theme_mode: str):
    if theme_mode == "Dark":
        plt.rcParams.update({
            "figure.facecolor": "#111827",
            "axes.facecolor": "#111827",
            "savefig.facecolor": "#111827",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e5eefc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#e5eefc",
            "axes.titlecolor": "#f8fafc",
            "grid.color": "#334155",
            "legend.facecolor": "#111827",
            "legend.edgecolor": "#334155",
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.edgecolor": "#d1d9e6",
            "axes.labelcolor": "#102033",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "text.color": "#102033",
            "axes.titlecolor": "#102033",
            "grid.color": "#dbe4f0",
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d1d9e6",
        })


# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------

def init_state():
    defaults = {
        "file_hash": None,
        "raw_df": None,
        "bundle": None,
        "uplift_results": None,
        "hte_results": None,
        "siamese_results": None,
        "appearance": "Light",
        "font_scale": 1.30,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)



def reset_analysis_state(new_hash: Optional[str]):
    st.session_state.file_hash = new_hash
    st.session_state.uplift_results = None
    st.session_state.hte_results = None
    st.session_state.siamese_results = None


def get_bundled_dataset_map() -> dict[str, str]:
    return {
        "Original dataset": "ab_testing.csv",
        "Modified dataset": "ab_testing_modified.csv",
    }


def load_bundled_dataset(dataset_label: str) -> pd.DataFrame:
    dataset_map = get_bundled_dataset_map()
    file_name = dataset_map[dataset_label]
    file_path = Path(__file__).resolve().parent / file_name
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_name}. Please place it in the same folder as app.py."
        )
    return pd.read_csv(file_path)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def plot_qini_curves(uplift_results: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, payload in uplift_results["qini_lines"].items():
        ax.plot(
            payload["fraction"],
            payload["qini"],
            label=f"{name} (AUQC={payload['auqc']:.3f})",
        )
    ax.plot(
        uplift_results["qini_random"]["fraction"],
        uplift_results["qini_random"]["qini"],
        linestyle="--",
        label="Random targeting baseline",
    )
    ax.set_xlabel("Fraction of users targeted")
    ax.set_ylabel("Qini incremental gain")
    ax.set_title("Qini Curves on the Test Set")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig



def plot_bootstrap_distribution(values, title, xlabel):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    lo = np.quantile(values, 0.025)
    hi = np.quantile(values, 0.975)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.histplot(values, bins=35, kde=True, ax=ax, alpha=0.55)
    ax.axvspan(lo, hi, alpha=0.15, label="95% CI")
    ax.axvline(mean, linewidth=2.2, label=f"Mean = {mean:.4f}")
    ax.axvline(lo, linestyle="--", linewidth=1.8)
    ax.axvline(hi, linestyle="--", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig



def plot_bootstrap_metric_by_model(boot_df: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.kdeplot(data=boot_df, x=metric, hue="model", common_norm=False, linewidth=2, ax=ax)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap Distribution of {metric.replace('_', ' ').title()}")
    fig.tight_layout()
    return fig



def plot_segment_bars(tbl: Optional[pd.DataFrame], seg_col: str, title: str):
    if tbl is None or tbl.empty:
        return None
    plot_tbl = tbl.dropna(subset=["uplift"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(plot_tbl[seg_col].astype(str), plot_tbl["uplift"])
    ax.set_xlabel(seg_col)
    ax.set_ylabel("Observed uplift")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig



def plot_history_curves(history: dict, experiment_title: str):
    figs = {}

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax1.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{experiment_title}: Epoch vs Loss")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    figs["loss"] = fig1

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(history["epoch"], history["train_auc"], label="Train AUC")
    ax2.plot(history["epoch"], history["val_auc"], label="Validation AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title(f"{experiment_title}: Epoch vs AUC")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    figs["auc"] = fig2

    if "train_acc" in history and "val_acc" in history:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(history["epoch"], history["train_acc"], label="Train accuracy")
        ax3.plot(history["epoch"], history["val_acc"], label="Validation accuracy")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.set_title(f"{experiment_title}: Epoch vs Accuracy")
        ax3.legend(frameon=False)
        fig3.tight_layout()
        figs["acc"] = fig3

    return figs



def plot_roc(fpr, tpr, auc_value: float, title: str):
    fig, ax = plt.subplots(figsize=(6.3, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Narrative helpers
# -----------------------------------------------------------------------------

def interpret_uplift_results(bundle: dict, uplift_results: dict) -> str:
    top = uplift_results["summary_df"].iloc[0]
    ate = bundle["ate_test"]
    direction = "positive" if ate > 0 else "negative" if ate < 0 else "neutral"
    return (
        f"The test-set average treatment effect is {ate:.4f}, indicating an overall {direction} "
        f"difference between treatment B and control A. Under the selected top-k policy "
        f"({uplift_results['manual_topk']:.0%}), the strongest uplift model in this run is "
        f"{top['model']}, with IPS policy value {top['IPS_test_policy']:.4f} and AUQC {top['AUQC_test']:.4f}."
    )



def interpret_hte_results(hte_results: dict) -> str:
    interaction_df = hte_results["interaction_df"]
    if interaction_df.empty:
        return "No interaction terms were returned, so the HTE screen did not highlight any treatment-by-feature effects in this run."
    top = interaction_df.iloc[0]
    return (
        f"The most prominent interaction term in the exploratory logistic model is {top['term']} "
        f"with odds ratio {top['odds_ratio']:.4f} and p-value {top['p_value']:.4f}. "
        "These results should be read as exploratory subgroup signals rather than definitive causal subgroup effects."
    )



def interpret_siamese_results(siamese_summary: pd.DataFrame) -> str:
    if siamese_summary.empty:
        return "No Siamese experiments have been run yet."
    top = siamese_summary.iloc[0]
    qualitative = "close to random discrimination" if top["test_auc"] < 0.55 else "clearer pair discrimination"
    return (
        f"Among the Siamese experiments run in the app, {top['title']} achieved the highest test AUC "
        f"at {top['test_auc']:.4f}. This suggests {qualitative} on the current dataset."
    )


# -----------------------------------------------------------------------------
# UI rendering
# -----------------------------------------------------------------------------

def render_overview():
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin-bottom:0.35rem;">ML-Driven A/B Testing for Enhanced Digital Ad Optimization</h2>
            <p style="margin-bottom:0.25rem;">
                This dashboard presents the core analytical components of the final year project through an interactive interface.
                It combines uplift modelling, exploratory heterogeneous treatment effect analysis, and Siamese-network experiments
                within a single platform designed to showcase the project cohesively.
            </p>
            <p class="small-note">Choose a dataset in the sidebar, then run the analytical components you would like to present.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="section-card">
            <h4>1. Data & Experiment Setup</h4>
            <p>Inspect the A/B dataset, treatment mapping, outcome mapping, and the train / validation / test split.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
            <h4>2. Uplift & HTE Analysis</h4>
            <p>Compare T-learner uplift models, policy values, Qini curves, bootstrap intervals, and subgroup uplift patterns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="section-card">
            <h4>3. Siamese Experiments</h4>
            <p>Present the baseline pair-based CNN and the bag-based CNN using the supplied project modules and configurations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_dataset_section(bundle: dict):
    st.subheader("Dataset and Experimental Design")
    df = bundle["df"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Train / Val / Test", f"{len(bundle['X_train'])} / {len(bundle['X_val'])} / {len(bundle['X_test'])}")
    m4.metric("Test ATE (B - A)", f"{bundle['ate_test']:.4f}")

    tab1, tab2, tab3 = st.tabs(["Preview", "Schema", "Split Summary"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        schema = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(x) for x in df.dtypes],
                "missing_values": df.isna().sum().values,
            }
        )
        st.dataframe(schema, use_container_width=True)

    with tab3:
        split_df = pd.DataFrame(
            {
                "split": ["Train", "Validation", "Test"],
                "size": [len(bundle["X_train"]), len(bundle["X_val"]), len(bundle["X_test"])],
                "conversion_rate": [
                    float(bundle["y_train"].mean()),
                    float(bundle["y_val"].mean()),
                    float(bundle["y_test"].mean()),
                ],
                "treatment_rate": [
                    float(bundle["T_train"].mean()),
                    float(bundle["T_val"].mean()),
                    float(bundle["T_test"].mean()),
                ],
            }
        )
        st.dataframe(split_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Train conversion rate A", f"{bundle['rate_A_train']:.4f}")
            st.metric("Train conversion rate B", f"{bundle['rate_B_train']:.4f}")
        with c2:
            st.metric("Test conversion rate A", f"{bundle['rate_A_test']:.4f}")
            st.metric("Test conversion rate B", f"{bundle['rate_B_test']:.4f}")



def render_uplift_section(bundle: dict, uplift_results: Optional[dict]):
    st.subheader("Uplift Modelling")
    if uplift_results is None:
        st.info("Run the uplift/HTE block from the sidebar to populate this section.")
        return

    st.markdown(interpret_uplift_results(bundle, uplift_results))

    top = uplift_results["summary_df"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best uplift model", top["model"])
    c2.metric("Best IPS policy value", f"{top['IPS_test_policy']:.4f}")
    c3.metric("Best AUQC", f"{top['AUQC_test']:.4f}")
    c4.metric("Bootstrap ATE CI", uplift_results["ate_bootstrap_summary"]["formatted"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Summary",
        "Qini Curves",
        "Bootstrap",
        "Validation Top-k Tables",
    ])

    with tab1:
        st.dataframe(uplift_results["summary_df"], use_container_width=True)
        st.download_button(
            "Download uplift model summary",
            uplift_results["summary_df"].to_csv(index=False),
            file_name="uplift_model_summary.csv",
            mime="text/csv",
        )

        pred_means_df = pd.DataFrame(uplift_results["pred_means"]).T.reset_index().rename(columns={"index": "model"})
        st.markdown("**Predicted response summaries**")
        st.dataframe(pred_means_df, use_container_width=True)

    with tab2:
        st.pyplot(plot_qini_curves(uplift_results), use_container_width=True)

    with tab3:
        b1, b2 = st.columns(2)
        with b1:
            st.pyplot(
                plot_bootstrap_distribution(
                    uplift_results["ates_boot"],
                    "Bootstrap Distribution of ATE on Test",
                    "ATE",
                ),
                use_container_width=True,
            )
        with b2:
            st.pyplot(
                plot_bootstrap_metric_by_model(uplift_results["boot_df"], "mean_uplift"),
                use_container_width=True,
            )

        st.markdown("**Bootstrap policy summary**")
        st.dataframe(uplift_results["bootstrap_df"], use_container_width=True)

        selected_model = st.selectbox(
            "Inspect bootstrap distributions for a specific uplift model",
            list(uplift_results["bootstrap_metrics_by_model"].keys()),
            key="bootstrap_model_select",
        )
        metrics = uplift_results["bootstrap_metrics_by_model"][selected_model]
        b3, b4 = st.columns(2)
        with b3:
            st.pyplot(
                plot_bootstrap_distribution(
                    metrics["ips_vals"],
                    f"Bootstrap IPS Policy Value — {selected_model}",
                    "IPS policy value",
                ),
                use_container_width=True,
            )
        with b4:
            st.pyplot(
                plot_bootstrap_distribution(
                    metrics["auqc_vals"],
                    f"Bootstrap AUQC — {selected_model}",
                    "AUQC",
                ),
                use_container_width=True,
            )

    with tab4:
        model_name = st.selectbox(
            "Choose model",
            list(uplift_results["val_topk_tables"].keys()),
            key="topk_table_model_select",
        )
        st.dataframe(uplift_results["val_topk_tables"][model_name], use_container_width=True)



def render_glm_summary_tables(hte_results: dict):
    glm_model = hte_results.get("glm_model")

    if glm_model is None:
        st.code(hte_results["glm_summary_text"])
        return

    try:
        summary2 = glm_model.summary2()

        model_fit_df = summary2.tables[0].copy()
        coef_df = summary2.tables[1].reset_index().rename(columns={"index": "term"}).copy()

        for df_ in [model_fit_df, coef_df]:
            num_cols = df_.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                df_.loc[:, num_cols] = df_.loc[:, num_cols].round(4)

        st.markdown("**Model fit summary**")
        st.dataframe(model_fit_df, use_container_width=True)

        st.markdown("**Coefficient table**")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    except Exception:
        st.code(hte_results["glm_summary_text"])



def render_hte_section(hte_results: Optional[dict]):
    st.subheader("Exploratory Heterogeneous Treatment Effect Analysis")
    if hte_results is None:
        st.info("Run the uplift/HTE block from the sidebar to populate this section.")
        return

    st.markdown(interpret_hte_results(hte_results))

    interaction_df = hte_results["interaction_df"]
    c1, c2, c3 = st.columns(3)
    c1.metric("HTE sample size", f"{len(hte_results['df_hte']):,}")
    c2.metric("Interaction terms", len(interaction_df))
    c3.metric(
        "Lowest interaction p-value",
        f"{interaction_df['p_value'].min():.4f}" if not interaction_df.empty else "N/A",
    )

    tab1, tab2, tab3 = st.tabs(["Interaction Terms", "Segment Uplift Tables", "Visuals"])

    with tab1:
        st.dataframe(interaction_df, use_container_width=True)
        with st.expander("View full logistic regression summary"):
            render_glm_summary_tables(hte_results)

    with tab2:
        st.markdown("**Device**")
        st.dataframe(hte_results["device_uplift"], use_container_width=True)
        st.markdown("**Location**")
        st.dataframe(hte_results["location_uplift"], use_container_width=True)
        if hte_results["pv_uplift"] is not None:
            st.markdown("**Page Views quantiles**")
            st.dataframe(hte_results["pv_uplift"], use_container_width=True)
        if hte_results["ts_uplift"] is not None:
            st.markdown("**Time Spent quantiles**")
            st.dataframe(hte_results["ts_uplift"], use_container_width=True)

    with tab3:
        v1, v2 = st.columns(2)
        with v1:
            fig = plot_segment_bars(hte_results["device_uplift"], "Device", "Observed uplift by Device")
            if fig:
                st.pyplot(fig, use_container_width=True)
            fig = plot_segment_bars(hte_results["pv_uplift"], "PageViews_bin", "Observed uplift by Page Views quantile")
            if fig:
                st.pyplot(fig, use_container_width=True)
        with v2:
            fig = plot_segment_bars(hte_results["location_uplift"], "Location", "Observed uplift by Location")
            if fig:
                st.pyplot(fig, use_container_width=True)
            fig = plot_segment_bars(hte_results["ts_uplift"], "TimeSpent_bin", "Observed uplift by Time Spent quantile")
            if fig:
                st.pyplot(fig, use_container_width=True)



def render_siamese_section(siamese_results: Optional[dict]):
    st.subheader("Siamese Network Experiments")
    if not siamese_results:
        st.info("Run one or both Siamese experiments from the sidebar to populate this section.")
        return

    siamese_summary = summarise_siamese_results(siamese_results)
    st.markdown(interpret_siamese_results(siamese_summary))
    st.dataframe(siamese_summary, use_container_width=True)

    selected_experiment = st.selectbox(
        "Choose experiment",
        list(siamese_results.keys()),
        format_func=lambda x: siamese_results[x]["title"],
    )
    res = siamese_results[selected_experiment]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC", f"{res['test']['auc']:.4f}")
    c2.metric("Test accuracy", f"{res['test']['acc']:.4f}")
    c3.metric("Best validation AUC", f"{max(res['history']['val_auc']):.4f}")
    c4.metric("Epochs run", len(res['history']['epoch']))

    st.caption(f"Device used: {res['device']} | Encoded shapes: {res['encoded_shapes']}")

    tab1, tab2, tab3 = st.tabs(["Training Curves", "ROC Curves", "Configuration"])
    with tab1:
        figs = plot_history_curves(res["history"], res["title"])
        st.pyplot(figs["loss"], use_container_width=True)
        st.pyplot(figs["auc"], use_container_width=True)
        if "acc" in figs:
            st.pyplot(figs["acc"], use_container_width=True)

    with tab2:
        if "val" in res and "fpr" in res["val"]:
            st.pyplot(
                plot_roc(res["val"]["fpr"], res["val"]["tpr"], res["val"]["auc"], f"Validation ROC — {res['title']}"),
                use_container_width=True,
            )
        st.pyplot(
            plot_roc(res["test"]["fpr"], res["test"]["tpr"], res["test"]["auc"], f"Test ROC — {res['title']}"),
            use_container_width=True,
        )
        st.json(res["test"]["score_summary"])

    with tab3:
        st.json(res["config"])



def render_final_highlights(bundle: Optional[dict], uplift_results: Optional[dict], hte_results: Optional[dict], siamese_results: Optional[dict]):
    st.subheader("Presentation Highlights")
    points = []

    if bundle is not None:
        points.append(
            f"The dataset is split into train, validation, and test sets with preserved treatment-outcome stratification, and the test-set ATE is {bundle['ate_test']:.4f}."
        )
    if uplift_results is not None:
        top = uplift_results["summary_df"].iloc[0]
        points.append(
            f"Within the uplift block, {top['model']} gives the strongest test policy value at {top['IPS_test_policy']:.4f}, with AUQC {top['AUQC_test']:.4f}."
        )
    if hte_results is not None and not hte_results["interaction_df"].empty:
        top_term = hte_results["interaction_df"].iloc[0]
        points.append(
            f"The exploratory HTE screen highlights {top_term['term']} as the most notable interaction term, with p-value {top_term['p_value']:.4f}."
        )
    if siamese_results:
        siamese_summary = summarise_siamese_results(siamese_results)
        top_si = siamese_summary.iloc[0]
        points.append(
            f"For the Siamese experiments, {top_si['title']} produces the highest test AUC at {top_si['test_auc']:.4f}."
        )

    if not points:
        st.info("Upload the dataset and run at least one analysis block to generate presentation highlights.")
        return

    for i, point in enumerate(points, start=1):
        st.markdown(f"**{i}.** {point}")


# -----------------------------------------------------------------------------
# Main app flow
# -----------------------------------------------------------------------------

def main():
    init_state()

    with st.sidebar:
        st.header("Controls")
        st.radio(
            "Appearance",
            ["Light", "Dark"],
            horizontal=True,
            key="appearance",
            help="Switch the dashboard palette between light and dark presentation modes.",
        )
        if "font_scale" not in st.session_state:
            st.session_state.font_scale = 1.20
            
        st.slider(
            "Font size scale",
            min_value=1.00,
            max_value=1.80,
            step=0.05,
            key="font_scale",
            help="Increase text size for easier readability during presentation.",
        )
        st.divider()
        dataset_label = st.selectbox(
            "Dataset",
            options=list(get_bundled_dataset_map().keys()),
            help="Choose which bundled dataset to use throughout the dashboard.",
        )
        topk = st.slider("Manual top-k policy", 0.05, 1.00, 0.80, 0.05)
        boot_n = st.select_slider("Bootstrap samples", options=[200, 400, 800, 1000, 2000], value=2000)
        siamese_fast_mode = st.toggle("Fast demo mode for Siamese training", value=True)
        siamese_choices = st.multiselect(
            "Siamese experiments to run",
            options=["baseline_cnn_pair", "bag_cnn"],
            default=["baseline_cnn_pair", "bag_cnn"],
        )
        page = st.radio(
            "Go to section",
            [
                "Overview",
                "Dataset & Split",
                "Uplift Modelling",
                "HTE Analysis",
                "Siamese Networks",
            ],
        )

        run_uplift_hte = st.button("Run uplift + HTE", use_container_width=True)
        run_siamese = st.button("Run Siamese experiments", use_container_width=True)

    apply_theme_css(st.session_state.appearance, float(st.session_state.font_scale))
    apply_matplotlib_theme(st.session_state.appearance)
    render_overview()

    try:
        dataset_file_name = get_bundled_dataset_map()[dataset_label]
        file_hash = dataset_file_name

        if st.session_state.file_hash != file_hash:
            reset_analysis_state(file_hash)
            st.session_state.raw_df = load_bundled_dataset(dataset_label)
            st.session_state.bundle = prepare_project_data(st.session_state.raw_df)

        bundle = st.session_state.bundle
        st.success(f"Dataset loaded successfully: {dataset_label}.")
    except FileNotFoundError as e:
        bundle = None
        st.error(str(e))

    if bundle is not None and run_uplift_hte:
        with st.spinner("Running uplift models and HTE analysis..."):
            progress_box = st.empty()
            uplift_results = run_t_learner_analysis(
                bundle,
                manual_topk=topk,
                n_boot=int(boot_n),
                progress_callback=lambda msg: progress_box.info(msg),
            )
            hte_results = run_hte_analysis(bundle)
            progress_box.empty()
            st.session_state.uplift_results = uplift_results
            st.session_state.hte_results = hte_results

    if bundle is not None and run_siamese:
        with st.spinner("Running Siamese experiments..."):
            siamese_results = st.session_state.siamese_results or {}
            for exp in siamese_choices:
                if exp == "baseline_cnn_pair":
                    siamese_results[exp] = run_baseline_siamese(bundle, fast_mode=siamese_fast_mode)
                elif exp == "bag_cnn":
                    siamese_results[exp] = run_bag_siamese(bundle, fast_mode=siamese_fast_mode)
            st.session_state.siamese_results = siamese_results

    uplift_results = st.session_state.uplift_results
    hte_results = st.session_state.hte_results
    siamese_results = st.session_state.siamese_results

    if page == "Overview":
        if bundle is not None:
            st.markdown("### Current run snapshot")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(bundle['df']):,}")
            c2.metric("Test ATE", f"{bundle['ate_test']:.4f}")
            c3.metric(
                "Best uplift IPS",
                f"{uplift_results['summary_df'].iloc[0]['IPS_test_policy']:.4f}" if uplift_results is not None else "Not run",
            )
            if siamese_results:
                si = summarise_siamese_results(siamese_results).iloc[0]
                c4.metric("Best Siamese test AUC", f"{si['test_auc']:.4f}")
            else:
                c4.metric("Best Siamese test AUC", "Not run")
    elif page == "Dataset & Split":
        if bundle is not None:
            render_dataset_section(bundle)
    elif page == "Uplift Modelling":
        render_uplift_section(bundle, uplift_results) if bundle is not None else st.info("Dataset file not found.")
    elif page == "HTE Analysis":
        render_hte_section(hte_results) if bundle is not None else st.info("Dataset file not found.")
    elif page == "Siamese Networks":
        render_siamese_section(siamese_results) if bundle is not None else st.info("Dataset file not found.")


if __name__ == "__main__":
    main()
