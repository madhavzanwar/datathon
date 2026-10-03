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
│   ├── train_cat.py             <- CatBoost multi-class model trainer
│   ├── optimize_thresholds.py   <- Nelder-Mead class probability threshold optimizer
│   ├── ensemble.py              <- Out-of-fold blending & rank-weighted ensemble
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
| **LightGBM (Engineered Feats)** | `0.812490` | `0.816383` | ✅ Completed |
| **XGBoost Hist (Engineered Feats)** | `0.813150` | `0.817221` | ✅ Completed |
| **Ensemble Blend (LGB + XGB + Thresholds)** | `0.814980` | **`0.818443`** | ✅ Top Submission |

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

### 4. Train Models
```bash
python src/train_lgb.py
python src/train_xgb.py
python src/train_cat.py
```

### 5. Run Ensembling & Threshold Optimization
```bash
python src/ensemble.py
```
This automatically produces a verified submission at `outputs/submissions/ensemble_submission.csv`.

---

## 🎯 5. Action Items & Roadmap for Teammate

1. **CatBoost & TabNet Integration**:
   - Run `python src/train_cat.py` and tune depth/l2 regularization to add tree diversity.
   - Build a PyTorch ResNet / TabNet tabular neural network in `src/train_nn.py` for model family diversity.
2. **Feature Engineering Iterations**:
   - Cluster-based distance features (k-means cluster centers on imputed train folds).
   - Frequency encoding / target encoding on binned continuous columns.
3. **Class-Wise Error Reduction**:
   - Over 50% of errors occur between Class 5 and Class 6. Build a binary specialist cascade classifier for Class 5 vs Class 6 ambiguity resolution.
4. **Ensemble Stacking**:
   - Train a Logistic Regression / Ridge meta-learner on out-of-fold probability vectors.
