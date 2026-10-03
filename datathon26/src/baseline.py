import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import seed_everything, compute_macro_f1, validate_submission

def train_baseline_lgb():
    print("=" * 70)
    print("      MILESTONE 2: LIGHTGBM 5-FOLD BASELINE PIPELINE")
    print("=" * 70)
    
    seed_everything(42)
    
    # 1. Load Data
    print("\n[1/5] Loading Data and Folds...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    folds_df = pd.read_csv("data/folds.csv")
    
    train_df['fold'] = folds_df['fold']
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'fold']]
    print(f"Features: {len(features)} numerical features")
    
    X_test = test_df[features].values
    test_ids = test_df['id'].values
    
    y_train = train_df['target'].values - 1 # map 1..7 to 0..6
    
    n_classes = 7
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    
    # 2. LightGBM Hyperparameters
    lgb_params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
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
    
    print("\n[2/5] Training 5-Fold LightGBM...")
    fold_f1_scores = []
    
    for fold in range(5):
        print(f"\n>>> Fold {fold + 1}/5")
        trn_mask = (train_df['fold'] != fold)
        val_mask = (train_df['fold'] == fold)
        
        X_trn = train_df.loc[trn_mask, features].values
        y_trn = y_train[trn_mask]
        
        X_val = train_df.loc[val_mask, features].values
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
        fold_f1 = compute_macro_f1(train_df.loc[val_mask, 'target'].values, val_preds_cls, verbose=False)
        fold_f1_scores.append(fold_f1)
        print(f"  Fold {fold + 1} Best Iteration: {model.best_iteration_} | Macro F1: {fold_f1:.6f}")
        
        # Test predictions
        test_probs += model.predict_proba(X_test) / 5.0
        
        del X_trn, y_trn, X_val, y_val, model
        gc.collect()
        
    print("\n" + "=" * 70)
    print(f"Mean 5-Fold Macro F1 (Argmax): {np.mean(fold_f1_scores):.6f} (+/- {np.std(fold_f1_scores):.6f})")
    print("=" * 70)
    
    # 3. Full OOF Evaluation
    oof_preds_cls = np.argmax(oof_probs, axis=1) + 1
    overall_f1 = compute_macro_f1(train_df['target'].values, oof_preds_cls, verbose=True)
    
    # 4. Save OOF & Test Probabilities
    np.save("outputs/oof/oof_lgb_baseline.npy", oof_probs)
    np.save("outputs/test_preds/preds_lgb_baseline.npy", test_probs)
    print("\n[4/5] Saved OOF and Test probability arrays.")
    
    # 5. Generate and Validate Submission
    print("\n[5/5] Generating Submission File...")
    sub_df = pd.DataFrame({
        'id': test_ids,
        'target': np.argmax(test_probs, axis=1) + 1
    })
    sub_path = "outputs/submissions/sub_lgb_baseline.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"  -> Saved submission to {sub_path}")
    
    validate_submission(sub_df)
    print("\n" + "=" * 70)
    print(f"BASELINE COMPLETED! Overall OOF Macro F1: {overall_f1:.6f}")
    print("=" * 70)
    
    return overall_f1

if __name__ == "__main__":
    train_baseline_lgb()
