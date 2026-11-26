# Recall Prediction for Spaced Repetition Learning

English · [简体中文](README.zh-CN.md)

A data science study of how model architecture, prediction targets, and learning-history features affect recall-rate prediction. The project compares twelve models on Duolingo learning traces and investigates the differences behind their evaluation scores.

[Final report](deliverables/ELEC3544_Group4_final_report.pdf) · [Presentation](deliverables/final_presentation.pptx) · [Notebook guide](NOTEBOOKS.md)

## Contents

- [1. Project overview](#1-project-overview)
- [2. Results and findings](#2-results-and-findings)
- [3. Quick start](#3-quick-start)
- [4. Repository structure](#4-repository-structure)
- [5. References and team](#5-references-and-team)

## 1. Project overview

### 1.1 Background and motivation

Spaced repetition schedules reviews over time to support long-term retention. Predicting how well a learner will recall an item provides a basis for choosing when to review it. Half-Life Regression (HLR) represents memory through a learned half-life and combines it with the elapsed review interval to estimate recall probability.

Inspired by HLR and subsequent neural forgetting-curve models, this project asks how additional learner and language features, nonlinear models, and alternative prediction targets affect prediction quality. All twelve models use the same cleaned data and fixed evaluation split.

### 1.2 Dataset

The study uses Duolingo's public learning traces, released with Settles and Meeder (2016). Each row represents a learner's practice session for a lexeme. The target, `p_recall`, is the proportion of correct responses within that session, calculated as `session_correct / session_seen`.

The raw release contains **12,854,226 records and 12 fields**. These include practice history (`history_seen`, `history_correct`), elapsed time (`delta`), learner and lexeme identifiers, learning and interface languages, timestamps, and current-session outcomes.

| Data-quality check | Result |
| --- | --- |
| Exact duplicates | 81 removed, retaining the first occurrence |
| Cleaned records | 12,854,145 |
| Basic validity violations after deduplication | 0 |
| Records with historical-count inconsistencies | 101,466 retained and reported |

The cleaned file is split in its original row order into **Train 72% (9,254,984 records)**, **Validation 18% (2,313,746)**, and **Test 10% (1,285,415)**. Validation supports model selection and early stopping. Test supports final evaluation. The target distribution is highly concentrated near full recall.

### 1.3 Modeling approach

The base input is `[sqrt(1 + c), sqrt(1 + w), 1]`, where `c = history_correct` and `w = history_seen - history_correct`. The review interval is converted from seconds to days, `Δ = delta / 86400`.

Recall labels are clipped to `[0.0001, 0.9999]`. Half-life models use the derived target `h = clip(-Δ / log2(p), 15/1440, 274)` days and predict recall through the forgetting curve

$$
\hat p = 2^{-\Delta/\hat h}.
$$

Predictions use the same clipping bounds. The half-life target is inferred from the recall label. The model families are

| Family | Models | Prediction approach |
| --- | --- | --- |
| Baselines | HLR, HLR -lex, Train mean | Linear half-life estimation, or a constant training-set mean |
| HLR feature extensions | HLR-User, HLR-Lang, HLR-User+Lang | Add learner and/or language one-hot features to HLR |
| Neural HLR | N-HLR, N-HLR -lex | Four ReLU hidden units predict `log2(h)`, followed by the forgetting curve |
| Tree models | RF-h, RF-p, LGBM-h, LGBM-p | Predict `log2(h)` from history, or predict recall directly from history and `Δ` |

The `-lex` suffix removes lexeme features. In the notebooks, RF-h and LGBM-h are named `RF-logh (E1)` and `LGBM-logh (E3)`. Their direct-recall counterparts are E2 and E4.

Inputs use information available before the current session. Categorical vocabularies are learned from Train only. A timestamp check confirms that every Train record precedes every Test record. The main metrics are MAE, RMSE, and AUC, with AUC labels defined by `np.rint(p_target)`.

## 2. Results and findings

### 2.1 Five main observations

The table reproduces the final report's results on the same **1,285,415 Test records**. Lower MAE and RMSE and higher AUC indicate better performance. Bold values mark the best score in each column.

| Model | MAE ↓ | RMSE ↓ | AUC ↑ |
| --- | ---: | ---: | ---: |
| HLR | 0.128033 | 0.290877 | 0.537856 |
| HLR -lex | 0.128283 | 0.290873 | 0.537459 |
| Train mean | 0.174210 | 0.271138 | 0.500000 |
| HLR-User | 0.127069 | 0.290409 | 0.540654 |
| HLR-Lang | 0.127938 | 0.290823 | 0.537993 |
| HLR-User+Lang | 0.126819 | 0.290242 | 0.540222 |
| N-HLR | **0.118898** | 0.287320 | 0.539874 |
| N-HLR -lex | 0.118909 | 0.287335 | 0.540314 |
| RF-h | 0.164276 | 0.317893 | 0.549080 |
| RF-p | 0.170206 | **0.267520** | 0.604966 |
| LGBM-h | 0.164277 | 0.317891 | 0.549281 |
| LGBM-p | 0.170254 | 0.267539 | **0.605412** |

1. **Learner and language features provide modest gains.** HLR-User+Lang improves MAE from 0.128033 to 0.126819, while the individual feature extensions produce smaller changes.
2. **Neural HLR achieves the lowest MAE.** N-HLR reaches 0.118898. Its lexeme-free counterpart performs almost identically on this metric.
3. **RF and LightGBM behave similarly under the same prediction target.** Both their half-life and direct-recall variants have closely matched aggregate scores.
4. **The tree-model prediction approach changes the trade-off.** Direct-recall models improve RMSE and AUC relative to their half-life counterparts, with higher MAE. These variants also add `Δ` as an input, so this comparison changes both the target and the features.
5. **Model rankings depend on the metric.** N-HLR has the lowest MAE, RF-p the lowest RMSE, and LGBM-p the highest AUC. The evaluation priority therefore affects the preferred model.

### 2.2 Further analysis

**Does N-HLR improve because of nonlinearity?** A matched-training ablation compares the original HLR -lex, a linear HLR using the neural model's training procedure, and a four-ReLU N-HLR. All use the same lexeme-free inputs, with three paired seeds for the two aligned variants.

| Model | Test MAE |
| --- | ---: |
| Original HLR -lex | 0.128283 |
| Aligned-Linear-HLR -lex | 0.119907 ± 0.000016 |
| N-HLR -lex | 0.118903 ± 0.000053 |

Most of the improvement appears after aligning the loss and optimization procedure. Nonlinearity adds a further mean MAE improvement of **0.001004**, with the same direction across all three seeds. The two parameterizations have different function-level regularization effects, even with the same L2 coefficient, so this is a controlled comparison with a remaining regularization limitation. See [08: nonlinearity ablation](08_nonlinearity_ablation.ipynb).

**Why do MAE and RMSE rank models differently?** In Test, 83.603% of labels equal the upper clipping bound. HLR makes smaller errors on this large group, helping its MAE, but can substantially overestimate low-recall records. Squaring increases the contribution of larger errors relative to smaller ones, giving these records more influence on MSE and hence RMSE. Grouped MAE/MSE contributions explain the ranking reversal between HLR and RF-p. See [09: MAE and RMSE analysis](09_mae_rmse_analysis.ipynb).

**Why are RF and LightGBM so similar?** We first investigated model reuse, row misalignment, cache overwrite, and post-processing errors. Independent prediction reconstruction and comparisons of raw and clipped outputs ruled out the tested implementation problems. A plausible explanation is that both methods learn highly similar predictive patterns from the shared inputs and targets. Clipping further increases the apparent similarity of the half-life variants. This is a preliminary diagnosis, and the underlying phenomenon remains future work. See [10: RF and LightGBM similarity](10_rf_lightgbm_similarity.ipynb).

These findings describe the current offline split. Historical-count inconsistencies are retained, and learners may appear in both Train and Test. Broader evaluation by learner, features, and random seed is left for future work.

## 3. Quick start

### 3.1 Environment setup

Use **64-bit Python 3.11**. The experiments run on CPU. Full-data random-forest training requires substantial RAM, and data processing and model caches occupy several GB of disk space. Runtime depends on the machine and available caches.

Clone the repository and enter its root directory

```bash
git clone https://github.com/Horace564/HLR.git
cd HLR
python -m venv .venv
```

Activate the environment on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux

```bash
source .venv/bin/activate
```

Install the pinned dependencies and register a notebook kernel

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name 3544-project --display-name "Python (3544-project)"
```

### 3.2 Dataset preparation

Download `settles.acl16.learning_traces.13m.csv.gz` from the original Duolingo dataset, hosted on Harvard Dataverse. The original project also links to the release. [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) · [Duolingo HLR repository](https://github.com/duolingo/halflife-regression)

Create `data/raw/` and place the compressed file at the following path, keeping its filename and gzip compression

```text
HLR/
└── data/
    └── raw/
        └── settles.acl16.learning_traces.13m.csv.gz
```

The raw file is approximately 361 MiB. The notebooks create cleaned data, feature caches, audit summaries, model artifacts, and figures as needed. Data files, caches, and the generated `reports/` directory are excluded from Git.

### 3.3 Running the standalone notebook

Start JupyterLab from the repository root

```bash
jupyter lab
```

Open `legacy_full_pipeline.ipynb`, select **Python (3544-project)**, and choose **Restart Kernel and Run All Cells**. VS Code can use the same Python environment.

Root-level notebooks use English by default. The [Chinese standalone edition](notebooks/zh-CN/legacy_full_pipeline.ipynb) and all Chinese modular notebooks are grouped under `notebooks/zh-CN/`. Each notebook links to its other language edition. When launched from that subdirectory, it automatically locates the project root and shares the same data and generated outputs.

This notebook contains the complete data-processing, model-definition, tuning, training, and analysis code. It can also run in a separate folder containing just the notebook and the raw dataset at the path above, with the required packages installed. Its twelve main models train in the notebook kernel. Preprocessing and the three-seed ablation can reuse their generated caches.

### 3.4 Running the modular notebooks

The **eleven notebooks numbered 00–10** cover the same workflow in separate stages. Keep `experiment_pipeline.py`, `pipeline_cells.py`, and `pipeline_steps/` beside them. Open any stage with the registered kernel and run all its cells.

Each stage supports a fresh kernel. Missing upstream results are generated automatically, and matching model caches are reused. Starting with 07–10 on a fresh checkout can therefore trigger model training. Following 00–10 in order makes the initial run easier to inspect.

The detailed stage map and execution notes are in the [Notebook guide](NOTEBOOKS.md#english). Repeated fits can show small numerical differences, particularly in rank-based diagnostics.

Both language editions use one shared training implementation and the same cache identities. Changing the display language can reuse existing modular model results. The two standalone editions each embed the full implementation and can run independently.

## 4. Repository structure

```text
HLR/
├── README.md                         # English project overview
├── README.zh-CN.md                   # Chinese project overview
├── NOTEBOOKS.md                      # Bilingual notebook guide
├── requirements.txt                  # Python dependencies
├── legacy_full_pipeline.ipynb        # Standalone end-to-end workflow
├── 00_data_preparation.ipynb
├── 01_eda.ipynb
├── 02_hlr_baselines.ipynb
├── 03_hlr_experiments.ipynb
├── 04_random_forest.ipynb
├── 05_lightgbm.ipynb
├── 06_neural_hlr.ipynb
├── 07_model_comparison.ipynb
├── 08_nonlinearity_ablation.ipynb
├── 09_mae_rmse_analysis.ipynb
├── 10_rf_lightgbm_similarity.ipynb
├── experiment_pipeline.py            # Modular execution and result caching
├── pipeline_cells.py                 # Loads shared computation blocks
├── pipeline_steps/                   # Data preparation, training, and evaluation
├── notebooks/
│   └── zh-CN/                       # All 12 Chinese notebook editions
├── deliverables/                     # Final report and presentation
└── paper/                            # Reference paper copies
```

Running the workflow creates `data/processed/`, `data/interim/`, `data/artifacts/`, and `reports/`. These outputs are generated locally. The repository keeps the executed notebook outputs for inspection, while excluding generated files, intermediate drafts, and notebook-build scripts from version control.

There are twelve English notebooks in the root directory and twelve corresponding Chinese notebooks in `notebooks/zh-CN/`.

## 5. References and team

### Selected references

- Settles, B., & Meeder, B. (2016). *A Trainable Spaced Repetition Model for Language Learning*. ACL. [Paper](https://research.duolingo.com/papers/settles.acl16.pdf)
- Zaidi, A., Caines, A., Moore, R., Buttery, P., & Rice, A. (2020). *Adaptive Forgetting Curves for Spaced Repetition Language Learning*. AIED. [Paper copy](paper/Adaptive%20Forgetting%20Curves%20for%20Spaced.pdf)
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

### Team

ELEC3544, Fall 2026 · Group 4

| Member | Student ID | Contribution |
| --- | --- | ---: |
| Fu Jing | 3036067121 | 50% |
| Chen Mingjia | 3036068553 | 50% |

Both members contributed equally to idea development, data cleaning, exploratory data analysis, modeling, training, and results analysis. All members agree with this allocation. The final report includes the contribution and AI-tool-use statements.
