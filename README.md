# 🏆 DATATHON 26–27 — Round 1 Classification Pipeline

> **Grandmaster-Grade Solution Architecture for 7-Class Tabular Classification**  
> **Evaluation Metric**: Macro F1 Score across all 7 target classes  
> **Target Benchmark**: CV Macro F1 $\ge 0.87$

---

## 📌 1. Project Overview

- **Dataset**: 174 numerical features (`f1`–`f174`) with intentional ~15% uniform MCAR missingness and synthetic noise.
- **Dataset Size**:
  - `train.csv`: 228,039 rows $\times$ 176 columns (`id`, `f1`–`f174`, `target`)
  - `test.csv`: 97,731 rows $\times$ 175 columns (`id`, `f1`–`f174`)
- **Key Insight**: Severe class imbalance (74.7x ratio between most frequent Class 6 at 26.11% and rarest Class 7 at 0.35%).
- **Adversarial Validation**: 3-fold LightGBM AUC = **`0.5047`** (Train and Test distributions are perfectly matched; local 5-fold CV is 100% reliable).

---

## 📂 2. Repository Structure

```text
datathon26/
├── data/
│   ├── train.csv                <- Place raw training data here (gitignored)
│   ├── test.csv                 <- Place raw test data here (gitignored)
│   ├── sample_submission.csv    <- Submission template
│   └── processed/               <- Processed feather datasets (auto-generated)
├── outputs/
│   ├── eda_report.md            <- Complete EDA & Adversarial Validation findings
│   ├── folds.csv                <- 5-Fold Stratified CV assignments (Seed 42)
│   ├── oof/                     <- Out-of-fold probability predictions
│   ├── test_preds/              <- Fold-averaged test predictions
│   └── submissions/             <- Validated submission CSV files
├── src/
│   ├── cv.py                    <- Reusable 5-Fold CV framework
│   ├── thorough_eda.py          <- Comprehensive EDA & Adversarial Validation script
│   ├── features.py              <- Leak-free feature engineering pipeline
│   ├── experiment_features.py   <- Systematic ablation study runner
│   ├── train_lgb.py             <- LightGBM multi-class model trainer
│   ├── train_xgb.py             <- XGBoost Hist GBDT model trainer
│   ├── train_catboost_cv.py     <- CatBoost multi-class model trainer
│   ├── train_nn.py              <- PyTorch Tabular Neural Network trainer (3-Layer ResNet/MLP)
│   ├── train_class56_specialist.py <- Dedicated Class 5 vs 6 Binary Specialist Classifier
│   ├── apply_cascade.py         <- Class 5 vs 6 Cascade Probability Refinement Engine
│   ├── optimize_thresholds.py   <- Nelder-Mead class probability threshold optimizer
│   ├── stacking.py              <- Super Ensemble Multi-Model Stacking & Meta-Learner
│   ├── generate_top_submission.py <- Master submission generator script
│   └── utils.py                 <- Metric computation, seeding, submission validator
├── COMPETITION.md               <- Full competition specifications and rules
├── requirements.txt             <- Python dependencies
└── README.md                    <- Setup & onboarding guide
```

---

## 📊 3. Current Benchmark Progress

| Model / Pipeline | CV Macro F1 (Raw Argmax) | CV Macro F1 (Optimized Multipliers) | Status |
| :--- | :---: | :---: | :---: |
| **LightGBM Baseline (Raw 174 Feats)** | `0.801700` | `0.804820` | ✅ Completed |
| **CatBoost GBDT (Engineered Feats)** | `0.788452` | `0.808038` | ✅ Completed |
| **LightGBM GBDT (Engineered Feats)** | `0.813832` | `0.816383` | ✅ Completed |
| **XGBoost Hist (Engineered Feats)** | `0.814718` | `0.817221` | ✅ Completed |
| **Class 5 vs 6 Binary Specialist Cascade** | `0.815430` | `0.818932` | ✅ Completed |
| **PyTorch Tabular Neural Network (3-Layer)** | `0.802352` | **`0.820469`** | ✅ Top Single Model |
| **Grand Super Ensemble (NN + XGB + LGB + Cat)** | `0.818479` | **`0.820773`** | 🏆 **NEW BENCHMARK HIGH** |

---

## ⚡ 4. Getting Started & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Put Data Files in Place
Ensure `train.csv` and `test.csv` are inside `data/`.

### 3. Generate Features
```bash
python src/features.py
```

### 4. Train All 4 Model Families
```bash
python src/train_lgb.py
python src/train_xgb.py
python src/train_catboost_cv.py
python src/train_nn.py
python src/train_class56_specialist.py
```

### 5. Generate Grand Super Ensemble Submission
```bash
python src/generate_top_submission.py
```
This automatically produces the top verified submission file at:  
`outputs/submissions/sub_grand_super_ensemble_0.820773.csv`.

---

## 🎯 5. Action Items & Roadmap for Teammates

1. **Feature Engineering Iterations**:
   - Cluster-based distance features (k-means cluster centers on imputed train folds).
   - Target encoding / group interaction features on top correlated feature pairs.
2. **Second-Level Meta Stacking**:
   - Train a Gradient Boosting Meta-Learner (LightGBM/XGBoost on out-of-fold probability vectors).
