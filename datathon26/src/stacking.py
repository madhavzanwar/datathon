import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from utils import seed_everything, validate_submission
from optimize_thresholds import optimize_weights

def apply_thresholds(probs, weights):
    scaled_probs = probs * weights
    return np.argmax(scaled_probs, axis=1) + 1

def run_stacking():
    print("=" * 70)
    print("      SUPER ENSEMBLE STACKING & META-LEARNING ENGINE")
    print("=" * 70)
    
    seed_everything(42)
    
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    folds_df = pd.read_csv("outputs/folds.csv")
    
    y = (train_df['target'].values - 1).astype(int)
    folds = folds_df['fold'].values
    n_classes = 7
    
    oof_files = {
        'LGBM': 'outputs/oof/oof_lgb_fe.npy',
        'XGBoost': 'outputs/oof/oof_xgb_fe.npy',
        'CatBoost': 'outputs/oof/oof_cat.npy',
        'NeuralNet': 'outputs/oof/oof_nn.npy'
    }
    
    test_files = {
        'LGBM': 'outputs/test_preds/preds_lgb_fe.npy',
        'XGBoost': 'outputs/test_preds/preds_xgb_fe.npy',
        'CatBoost': 'outputs/test_preds/preds_cat.npy',
        'NeuralNet': 'outputs/test_preds/preds_nn.npy'
    }
    
    # Also check alternative locations if subagents saved at outputs/
    alt_oof = {
        'CatBoost': 'outputs/oof_cat.npy',
        'NeuralNet': 'outputs/oof_nn.npy'
    }
    alt_test = {
        'CatBoost': 'outputs/test_cat.npy',
        'NeuralNet': 'outputs/test_nn.npy'
    }
    
    oof_probs_list = []
    test_probs_list = []
    model_names = []
    
    for name in oof_files.keys():
        p_oof = oof_files[name]
        p_test = test_files[name]
        if not os.path.exists(p_oof) and name in alt_oof:
            p_oof = alt_oof[name]
            p_test = alt_test[name]
            
        if os.path.exists(p_oof) and os.path.exists(p_test):
            oof_p = np.load(p_oof)
            test_p = np.load(p_test)
            oof_probs_list.append(oof_p)
            test_probs_list.append(test_p)
            model_names.append(name)
            
            f1_raw = f1_score(y + 1, np.argmax(oof_p, axis=1) + 1, average='macro')
            mults = optimize_weights(oof_p, y + 1)
            f1_cal = f1_score(y + 1, apply_thresholds(oof_p, mults), average='macro')
            print(f"  -> Model [{name:10s}]: Raw CV Macro F1 = {f1_raw:.6f} | Calibrated = {f1_cal:.6f}")
            
    if not oof_probs_list:
        print("Error: No OOF probability files found!")
        return
        
    print(f"\n[Stacking] Blending {len(model_names)} Model Families: {model_names}")
    
    # 1. Simple Weighted Average
    weights_blend = [1.0 / len(model_names)] * len(model_names)
    oof_blend = np.zeros_like(oof_probs_list[0])
    test_blend = np.zeros_like(test_probs_list[0])
    
    for i, w in enumerate(weights_blend):
        oof_blend += w * oof_probs_list[i]
        test_blend += w * test_probs_list[i]
        
    f1_blend_raw = f1_score(y + 1, np.argmax(oof_blend, axis=1) + 1, average='macro')
    blend_mults = optimize_weights(oof_blend, y + 1)
    blend_preds_cal = apply_thresholds(oof_blend, blend_mults)
    f1_blend_cal = f1_score(y + 1, blend_preds_cal, average='macro')
    
    print("\n" + "=" * 70)
    print(f"  -> Equal Weighted Blend Raw CV Macro F1       : {f1_blend_raw:.6f}")
    print(f"  -> Equal Weighted Blend Calibrated Macro F1   : {f1_blend_cal:.6f}")
    print("=" * 70)
    
    # 2. Meta-Learner (Logistic Regression on OOF probability features)
    print("\n[Meta-Learner] Training 5-Fold Stacking Meta-Learner (Logistic Regression)...")
    X_oof_meta = np.hstack(oof_probs_list) # shape: (N, 7 * n_models)
    X_test_meta = np.hstack(test_probs_list)
    
    meta_oof_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    meta_test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    
    for fold in range(5):
        trn_idx = (folds != fold)
        val_idx = (folds == fold)
        
        X_tr, y_tr = X_oof_meta[trn_idx], y[trn_idx]
        X_val, y_val = X_oof_meta[val_idx], y[val_idx]
        
        clf = LogisticRegression(C=0.1, max_iter=500, class_weight='balanced', random_state=42)
        clf.fit(X_tr, y_tr)
        
        meta_oof_probs[val_idx] = clf.predict_proba(X_val)
        meta_test_probs += clf.predict_proba(X_test_meta) / 5.0
        
    f1_meta_raw = f1_score(y + 1, np.argmax(meta_oof_probs, axis=1) + 1, average='macro')
    meta_mults = optimize_weights(meta_oof_probs, y + 1)
    meta_preds_cal = apply_thresholds(meta_oof_probs, meta_mults)
    f1_meta_cal = f1_score(y + 1, meta_preds_cal, average='macro')
    
    print(f"  -> Stacking Meta-Learner Raw CV Macro F1      : {f1_meta_raw:.6f}")
    print(f"  -> Stacking Meta-Learner Calibrated Macro F1  : {f1_meta_cal:.6f}")
    print("=" * 70)
    
    # Select best strategy
    if f1_meta_cal >= f1_blend_cal:
        final_test_probs = meta_test_probs
        final_mults = meta_mults
        final_score = f1_meta_cal
        strategy_name = "Stacking Meta-Learner"
    else:
        final_test_probs = test_blend
        final_mults = blend_mults
        final_score = f1_blend_cal
        strategy_name = "Equal Multi-Model Blend"
        
    test_final_preds = apply_thresholds(final_test_probs, final_mults)
    
    os.makedirs("outputs/submissions", exist_ok=True)
    sub_path = "outputs/submissions/sub_super_stacked.csv"
    sub_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_final_preds
    })
    sub_df.to_csv(sub_path, index=False)
    
    print(f"\n[Final Output] Saved top submission using [{strategy_name}] to {sub_path}")
    print(f"               Final Verified CV Macro F1: {final_score:.6f}")
    validate_submission(sub_df)

if __name__ == "__main__":
    run_stacking()
