import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb
from utils import seed_everything

def run_cv():
    print("=" * 70)
    print("      5-FOLD STRATIFIED CV FRAMEWORK (BALANCED LIGHTGBM)")
    print("=" * 70)
    
    seed_everything(42)
    
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load Data
    print("\n[1/5] Loading Raw Data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    features = [c for c in train_df.columns if c not in ['id', 'target']]
    print(f"  -> Features: {len(features)} raw numerical columns")
    print(f"  -> Train shape: {train_df.shape}")
    print(f"  -> Test shape:  {test_df.shape}")
    
    # 2. Generate and Save 5-Fold Stratified Split
    print("\n[2/5] Creating 5-Fold Stratified Split (Seed 42)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        train_df.loc[val_idx, 'fold'] = fold
        
    folds_export_path = "outputs/folds.csv"
    train_df[['id', 'target', 'fold']].to_csv(folds_export_path, index=False)
    print(f"  -> Saved fold assignments to {folds_export_path}")
    
    X_test = test_df[features].values.astype(np.float32)
    y_train = (train_df['target'].values - 1).astype(int) # map 1..7 to 0..6
    
    n_classes = 7
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    
    # 3. LightGBM Baseline with class_weight='balanced'
    lgb_params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'class_weight': 'balanced',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': -1,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'min_child_samples': 25,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 1200,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print("\n[3/5] Training 5-Fold LightGBM (class_weight='balanced')...")
    fold_f1_scores = []
    
    for fold in range(5):
        print(f"\n>>> Fold {fold + 1}/5")
        trn_mask = (train_df['fold'] != fold)
        val_mask = (train_df['fold'] == fold)
        
        X_trn = train_df.loc[trn_mask, features].values.astype(np.float32)
        y_trn = y_train[trn_mask]
        
        X_val = train_df.loc[val_mask, features].values.astype(np.float32)
        y_val = y_train[val_mask]
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=200)
        ]
        
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        
        val_preds_prob = model.predict_proba(X_val)
        oof_probs[val_mask] = val_preds_prob
        
        val_preds_cls = np.argmax(val_preds_prob, axis=1) + 1
        y_val_orig = train_df.loc[val_mask, 'target'].values
        fold_f1 = f1_score(y_val_orig, val_preds_cls, average='macro', zero_division=0)
        fold_f1_scores.append(fold_f1)
        print(f"  Fold {fold + 1} Best Iteration: {model.best_iteration_} | Macro F1: {fold_f1:.6f}")
        
        test_probs += model.predict_proba(X_test) / 5.0
        
        del X_trn, y_trn, X_val, y_val, model
        gc.collect()
        
    print("\n" + "=" * 70)
    print(f"Mean 5-Fold Macro F1: {np.mean(fold_f1_scores):.6f} (+/- {np.std(fold_f1_scores):.6f})")
    print("=" * 70)
    
    # 4. Overall Out-of-Fold Evaluation & Diagnostics
    print("\n[4/5] Evaluating Full Out-of-Fold Predictions...")
    oof_preds_cls = np.argmax(oof_probs, axis=1) + 1
    y_true_orig = train_df['target'].values
    overall_macro_f1 = f1_score(y_true_orig, oof_preds_cls, average='macro', zero_division=0)
    
    print(f"\n==================================================")
    print(f"   OVERALL OOF MACRO F1 SCORE: {overall_macro_f1:.6f}")
    print(f"==================================================")
    
    print("\nDetailed Per-Class Classification Report:")
    report_str = classification_report(y_true_orig, oof_preds_cls, digits=4, zero_division=0)
    print(report_str)
    
    print("\nConfusion Matrix (Rows = Actual, Columns = Predicted):")
    cm = confusion_matrix(y_true_orig, oof_preds_cls, labels=list(range(1, 8)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual Class {i}" for i in range(1, 8)],
        columns=[f"Pred {i}" for i in range(1, 8)]
    )
    print(cm_df.to_string())
    
    # Analyze Weakest Classes
    per_class_f1 = f1_score(y_true_orig, oof_preds_cls, average=None, labels=list(range(1, 8)), zero_division=0)
    weakest_order = np.argsort(per_class_f1)
    
    print("\nClass Performance Ranked (Weakest to Strongest):")
    for rank, idx in enumerate(weakest_order, 1):
        cls = idx + 1
        print(f"  #{rank} Class {cls}: F1 = {per_class_f1[idx]:.4f} (Support: {(y_true_orig == cls).sum():,})")
        
    # 5. Save OOF and Test Predictions
    print("\n[5/5] Saving Probability Arrays...")
    oof_out_path = "outputs/oof_lgb.npy"
    test_out_path = "outputs/test_lgb.npy"
    
    np.save(oof_out_path, oof_probs)
    np.save(test_out_path, test_probs)
    print(f"  -> Saved OOF probabilities to {oof_out_path} ({oof_probs.shape})")
    print(f"  -> Saved Test probabilities to {test_out_path} ({test_probs.shape})")
    print("=" * 70)
    print("CV PIPELINE RUN COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_cv()
