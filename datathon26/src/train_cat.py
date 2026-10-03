import os
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from utils import seed_everything, compute_macro_f1, validate_submission
from optimize_thresholds import optimize_weights

def train_cat_with_features():
    print("=" * 70)
    print("      MILESTONE 4: CATBOOST 5-FOLD CLASSIFIER")
    print("=" * 70)
    
    seed_everything(42)
    
    # 1. Load Processed Features
    print("\n[1/5] Loading Processed Feature Datasets...")
    train_df = pd.read_feather("data/processed/train_features.feather")
    test_df = pd.read_feather("data/processed/test_features.feather")
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'fold']]
    print(f"Total Features (Base + Engineered): {len(features)}")
    
    X_test = test_df[features].values.astype(np.float32)
    test_ids = test_df['id'].values
    y_train = (train_df['target'].values - 1).astype(int)
    
    n_classes = 7
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    
    # 2. CatBoost Hyperparameters
    cat_params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'iterations': 1200,
        'learning_rate': 0.08,
        'depth': 6,
        'l2_leaf_reg': 4.0,
        'random_seed': 42,
        'thread_count': -1,
        'verbose': 200
    }
    
    print("\n[2/5] Training 5-Fold CatBoost...")
    fold_f1_scores = []
    
    for fold in range(5):
        print(f"\n>>> Fold {fold + 1}/5")
        trn_mask = (train_df['fold'] != fold)
        val_mask = (train_df['fold'] == fold)
        
        X_trn = train_df.loc[trn_mask, features].values.astype(np.float32)
        y_trn = y_train[trn_mask]
        
        X_val = train_df.loc[val_mask, features].values.astype(np.float32)
        y_val = y_train[val_mask]
        
        model = CatBoostClassifier(**cat_params, early_stopping_rounds=60)
        
        model.fit(
            X_trn, y_trn,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        val_preds_prob = model.predict_proba(X_val)
        oof_probs[val_mask] = val_preds_prob
        
        val_preds_cls = np.argmax(val_preds_prob, axis=1) + 1
        fold_f1 = compute_macro_f1(train_df.loc[val_mask, 'target'].values, val_preds_cls, verbose=False)
        fold_f1_scores.append(fold_f1)
        print(f"  Fold {fold + 1} Best Iteration: {model.get_best_iteration()} | Macro F1: {fold_f1:.6f}")
        
        test_probs += model.predict_proba(X_test) / 5.0
        
        del X_trn, y_trn, X_val, y_val, model
        gc.collect()
        
    print("\n" + "=" * 70)
    print(f"Mean 5-Fold CatBoost Macro F1 (Raw Argmax): {np.mean(fold_f1_scores):.6f} (+/- {np.std(fold_f1_scores):.6f})")
    print("=" * 70)
    
    # 3. Optimize Probability Weights on OOF
    print("\n[3/5] Threshold & Weight Optimization on OOF...")
    y_true_orig = train_df['target'].values
    weights = optimize_weights(oof_probs, y_true_orig)
    
    # 4. Save OOF & Test Probabilities
    np.save("outputs/oof/oof_cat_fe.npy", oof_probs)
    np.save("outputs/test_preds/preds_cat_fe.npy", test_probs)
    print("\n[4/5] Saved OOF and Test probability arrays.")
    
    # 5. Generate Calibrated Submission
    print("\n[5/5] Generating Calibrated Submission...")
    calibrated_test_preds = np.argmax(test_probs * weights, axis=1) + 1
    sub_df = pd.DataFrame({
        'id': test_ids,
        'target': calibrated_test_preds
    })
    sub_path = "outputs/submissions/sub_cat_fe_calibrated.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"  -> Saved submission to {sub_path}")
    validate_submission(sub_df)
    
    print("\n" + "=" * 70)
    print("CATBOOST WITH ENGINEERED FEATURES COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    train_cat_with_features()
