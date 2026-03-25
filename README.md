# FYP-S2: ML-Driven A/B Testing for Enhanced Digital Ad Optimisation

This repository contains the full codebase for my Final Year Project, which extends a conventional A/B testing framework with machine learning methods for uplift modelling, heterogeneous treatment effect analysis, and Siamese neural network experiments.

The project is based on a two-arm digital advertising A/B testing dataset and investigates whether machine learning can provide deeper insight beyond aggregate treatment comparison, particularly in identifying uplift, subgroup variation, and treatment-arm separability.

In addition to the main analysis workflow, this repository also includes a Streamlit web application for interactive presentation of the project findings.

---

## Project components

This project consists of four main parts:

### 1. Data preparation and exploratory analysis
The dataset is cleaned, inspected, encoded, and split into train, validation, and test sets. Preprocessing pipelines are used to prepare the data for machine learning analysis.

### 2. Uplift modelling
T-learners are used to estimate individual treatment effects and assess whether treatment targeting could improve decision-making beyond applying a single treatment to everyone.

### 3. Heterogeneous treatment effect (HTE) analysis
Interaction logistic regression and subgroup uplift summaries are used to examine whether treatment effects differ meaningfully across observed user characteristics such as device and location.

### 4. Siamese neural network experiments
Two CNN-based Siamese experiments are included to test whether the treatment arms can be meaningfully separated from the available feature space:

- **Experiment 1: Row CNN**
- **Experiment 2: Block CNN**

An additional modified-dataset diagnostic experiment is also used to show that weak performance on the original dataset is mainly due to the nature of the data rather than implementation failure.

---

## Repository structure

The repository currently contains the following files:

### Main notebook and datasets
- `FYP Main.ipynb` — main notebook containing the primary project workflow and analysis
- `ab_testing.csv` — original dataset used for the main analyses
- `ab_testing_modified.csv` — modified dataset used for the diagnostic experiment

### Streamlit application
- `app.py` — main Streamlit application
- `analysis_core.py` — shared backend logic for preprocessing, uplift modelling, HTE analysis, and experiment execution
- `.streamlit/` — Streamlit configuration files

### Siamese experiment modules
- `run_siamese_experiment.py` — master runner for Siamese experiments
- `exp_01_row_cnn.py` — Experiment 1: Row CNN
- `exp_02_block_cnn.py` — Experiment 2: Block CNN

### Siamese support modules
- `siamese_config.py` — experiment settings and configuration
- `siamese_data.py` — data preparation and pair/block construction
- `siamese_encoders.py` — encoder definitions
- `siamese_models.py` — Siamese model architectures
- `siamese_losses.py` — loss functions
- `siamese_trainers.py` — model training routines
- `siamese_eval.py` — evaluation functions and metrics
- `siamese_utils.py` — utility/helper functions

### Environment
- `requirements.txt` — required Python packages

---

## Main methods used

### Uplift modelling
The uplift section uses T-learners with separate models for control and treatment groups. The estimated uplift is computed as the difference between predicted treatment and control outcome probabilities.

Typical evaluation outputs include:
- uplift score distributions
- uplift by decile
- IPS policy value
- Qini curves
- AUQC
- bootstrap confidence intervals and bootstrap histograms

### HTE analysis
The HTE section includes:
- interaction logistic regression
- interpretation of treatment-feature interaction terms
- subgroup uplift comparison tables

### Siamese experiments
The Siamese component investigates whether user records from treatment A and treatment B can be distinguished through learned representations.

- **Row CNN** operates at the row level
- **Block CNN** groups rows into blocks to reduce instability and attempt to capture stronger signal

Typical outputs include:
- training and validation loss
- validation and test AUC
- accuracy
- ROC curves
- score distributions

---

## Streamlit app features

The Streamlit app was built as an interactive showcase of the project. It includes:

- dataset upload and inspection
- split summary
- uplift modelling results
- IPS, Qini, and AUQC outputs
- bootstrap visualisations
- HTE analysis
- Siamese experiment pages for:
  - Row CNN
  - Block CNN

The app is intended for academic presentation and demonstration of the project workflow.
