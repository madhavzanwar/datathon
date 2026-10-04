import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import extract_features
from src.utils import seed_everything, compute_macro_f1, validate_submission
from src.optimize_thresholds import optimize_weights

def run_catboost_cv():
    print("=" * 70)
    print("      CATBOOST 5-FOLD STRATIFIED CV PIPELINE")
    print("=" * 70)
    
    seed_everything(42)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/oof", exist_ok=True)
    os.makedirs("outputs/test_preds", exist_ok=True)
    os.makedirs("outputs/submissions", exist_ok=True)
    
    # 1. Load Datasets
    print("\n[1/5] Loading Raw Data and Folds...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    folds_df = pd.read_csv("outputs/folds.csv")
    
    train_df['fold'] = folds_df['fold']
    
    print(f"  -> Train shape: {train_df.shape}")
    print(f"  -> Test shape:  {test_df.shape}")
    print(f"  -> Folds distribution:\n{folds_df['fold'].value_counts().sort_index().to_string()}")
    
    n_classes = 7
    n_samples_train = len(train_df)
    n_samples_test = len(test_df)
    
    oof_probs = np.zeros((n_samples_train, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_samples_test, n_classes), dtype=np.float32)
    
    y_true = train_df['target'].values
    y_train_mapped = y_true - 1  # 0-indexed for CatBoost (0..6)
    
    # 2. CatBoost Parameters
    cat_params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'auto_class_weights': 'Balanced',
        'iterations': 1200,
        'learning_rate': 0.06,
        'depth': 6,
        'random_seed': 42,
        'thread_count': -1,
        'verbose': 200
    }
    
    print("\n[2/5] Training 5-Fold CatBoost (Strict CV Feature Extraction)...")
    print(f"CatBoost Parameters: {cat_params}")
    
    fold_f1_scores = []
    
    for fold in range(5):
        print(f"\n" + "=" * 50)
        print(f" >>> FOLD {fold + 1} / 5")
        print("=" * 50)
        
        trn_idx = train_df['fold'] != fold
        val_idx = train_df['fold'] == fold
        
        raw_trn = train_df[trn_idx].reset_index(drop=True)
        raw_val = train_df[val_idx].reset_index(drop=True)
        
        print(f"  Extracting features for Fold {fold + 1} (fitting imputer/scaler/SVD on train fold)...")
        trn_fe, svd_m, imp_m, scl_m = extract_features(raw_trn, is_train=True)
        val_fe, _, _, _ = extract_features(raw_val, is_train=False, svd_model=svd_m, imputer=imp_m, scaler=scl_m)
        test_fold_fe, _, _, _ = extract_features(test_df, is_train=False, svd_model=svd_m, imputer=imp_m, scaler=scl_m)
        
        feature_cols = [c for c in trn_fe.columns if c not in ['id', 'target', 'fold']]
        print(f"  -> Total features used: {len(feature_cols)}")
        
        X_trn = trn_fe[feature_cols].values.astype(np.float32)
        y_trn = (trn_fe['target'].values - 1).astype(int)
        
        X_val = val_fe[feature_cols].values.astype(np.float32)
        y_val = (val_fe['target'].values - 1).astype(int)
        
        X_test = test_fold_fe[feature_cols].values.astype(np.float32)
        
        model = CatBoostClassifier(**cat_params, early_stopping_rounds=60)
        
        model.fit(
            X_trn, y_trn,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        val_preds_prob = model.predict_proba(X_val)
        oof_probs[val_idx.values] = val_preds_prob
        
        val_preds_cls = np.argmax(val_preds_prob, axis=1) + 1
        fold_f1 = compute_macro_f1(raw_val['target'].values, val_preds_cls, verbose=False)
        fold_f1_scores.append(fold_f1)
        print(f"  -> Fold {fold + 1} Best Iteration: {model.get_best_iteration()} | Macro F1: {fold_f1:.6f}")
        
        test_probs += model.predict_proba(X_test) / 5.0
        
        del raw_trn, raw_val, trn_fe, val_fe, test_fold_fe, X_trn, y_trn, X_val, y_val, X_test, model, svd_m, imp_m, scl_m
        gc.collect()
        
    print("\n" + "=" * 70)
    print("PER-FOLD MACRO F1 RESULTS:")
    for i, score in enumerate(fold_f1_scores, 1):
        print(f"  Fold {i}: {score:.6f}")
    
    mean_fold_f1 = np.mean(fold_f1_scores)
    std_fold_f1 = np.std(fold_f1_scores)
    print(f"Mean Per-Fold Macro F1: {mean_fold_f1:.6f} (+/- {std_fold_f1:.6f})")
    
    raw_oof_preds = np.argmax(oof_probs, axis=1) + 1
    uncalibrated_macro_f1 = compute_macro_f1(y_true, raw_oof_preds, verbose=False)
    print(f"Overall Uncalibrated OOF Macro F1: {uncalibrated_macro_f1:.6f}")
    print("=" * 70)
    
    # 3. Save OOF and Test Probabilities
    print("\n[3/5] Saving OOF and Test Probabilities...")
    np.save("outputs/oof_cat.npy", oof_probs)
    np.save("outputs/test_cat.npy", test_probs)
    # Also save to outputs/oof/ and outputs/test_preds/ for modular compatibility
    np.save("outputs/oof/oof_cat.npy", oof_probs)
    np.save("outputs/test_preds/test_cat.npy", test_probs)
    print("  -> Saved outputs/oof_cat.npy")
    print("  -> Saved outputs/test_cat.npy")
    
    # 4. Threshold & Weight Calibration
    print("\n[4/5] Running Threshold Calibration via optimize_thresholds logic...")
    best_weights = optimize_weights(oof_probs, y_true)
    
    calibrated_oof_preds = np.argmax(oof_probs * best_weights, axis=1) + 1
    calibrated_macro_f1 = compute_macro_f1(y_true, calibrated_oof_preds, verbose=True)
    
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY:")
    print(f"  Uncalibrated 5-Fold OOF Macro F1 : {uncalibrated_macro_f1:.6f}")
    print(f"  Calibrated 5-Fold OOF Macro F1   : {calibrated_macro_f1:.6f}")
    print(f"  Improvement (Delta)               : {calibrated_macro_f1 - uncalibrated_macro_f1:+.6f}")
    print("=" * 70)
    
    # 5. Generate and Validate Calibrated Submission
    print("\n[5/5] Generating Calibrated Submission...")
    calibrated_test_preds = np.argmax(test_probs * best_weights, axis=1) + 1
    
    sub_df = pd.DataFrame({
        'id': test_df['id'].values,
        'target': calibrated_test_preds
    })
    
    sub_path = "outputs/submissions/sub_cat_calibrated.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"  -> Saved submission to {sub_path}")
    
    validate_submission(sub_df)
    
    print("\n" + "=" * 70)
    print("CATBOOST 5-FOLD CV & CALIBRATION SUCCESSFULLY COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    run_catboost_cv()
