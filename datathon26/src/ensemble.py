import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils import seed_everything, compute_macro_f1, validate_submission
from optimize_thresholds import optimize_weights

def build_ensemble():
    print("=" * 70)
    print("      MILESTONE 6: MULTI-MODEL ENSEMBLE & BLENDING HARNESS")
    print("=" * 70)
    
    seed_everything(42)
    
    folds_df = pd.read_csv("data/folds.csv")
    y_true = folds_df['target'].values
    
    sample_df = pd.read_csv("data/sample_submission.csv")
    test_ids = sample_df['id'].values
    
    # Check all available OOF predictions
    oof_files = sorted(glob.glob("outputs/oof/*.npy"))
    print(f"\nFound {len(oof_files)} OOF prediction arrays:")
    
    model_names = []
    oofs = []
    test_preds = []
    
    for f in oof_files:
        name = os.path.basename(f).replace("oof_", "").replace(".npy", "")
        test_file = f"outputs/test_preds/preds_{name}.npy"
        
        if os.path.exists(test_file):
            oof_prob = np.load(f)
            test_prob = np.load(test_file)
            
            raw_f1 = compute_macro_f1(y_true, np.argmax(oof_prob, axis=1) + 1, verbose=False)
            print(f"  -> Model [{name:20s}] | Raw Argmax OOF Macro F1: {raw_f1:.6f}")
            
            model_names.append(name)
            oofs.append(oof_prob)
            test_preds.append(test_prob)
            
    if len(oofs) == 0:
        print("No matching OOF and test prediction pairs found!")
        return
        
    # 1. Simple Equal-Weighted Average Blend
    print("\n--- Equal-Weighted Average Blend ---")
    avg_oof = np.mean(oofs, axis=0)
    avg_test = np.mean(test_preds, axis=0)
    
    raw_blend_f1 = compute_macro_f1(y_true, np.argmax(avg_oof, axis=1) + 1, verbose=False)
    print(f"  -> Raw Blend Argmax Macro F1: {raw_blend_f1:.6f}")
    
    # 2. Optimal Probability Thresholding on Equal Blend
    print("\n--- Calibrating Equal-Weighted Blend ---")
    weights_eq = optimize_weights(avg_oof, y_true)
    calibrated_test_1 = np.argmax(avg_test * weights_eq, axis=1) + 1
    
    sub1_path = "outputs/submissions/sub_final_1_blend_equal.csv"
    sub1_df = pd.DataFrame({'id': test_ids, 'target': calibrated_test_1})
    sub1_df.to_csv(sub1_path, index=False)
    print(f"  -> Saved Submission 1: {sub1_path}")
    validate_submission(sub1_df)
    
    # 3. Model-Weight Optimization + Threshold Calibration (Super Ensemble)
    if len(oofs) > 1:
        print("\n--- Optimizing Inter-Model Blending Weights ---")
        n_models = len(oofs)
        
        def model_blend_loss(model_weights):
            mw = np.array(model_weights)
            mw = np.maximum(mw, 0)
            mw = mw / (np.sum(mw) + 1e-9)
            blended = np.zeros_like(oofs[0])
            for i in range(n_models):
                blended += oofs[i] * mw[i]
            preds = np.argmax(blended, axis=1) + 1
            return -compute_macro_f1(y_true, preds, verbose=False)
            
        res_m = minimize(
            model_blend_loss,
            np.ones(n_models) / n_models,
            method='Nelder-Mead',
            options={'maxiter': 300, 'disp': True}
        )
        
        opt_mw = np.maximum(res_m.x, 0)
        opt_mw = opt_mw / np.sum(opt_mw)
        print("\nOptimal Model Blending Weights:")
        for name, w in zip(model_names, opt_mw):
            print(f"  - {name:20s}: {w*100:5.2f}%")
            
        opt_blended_oof = np.zeros_like(oofs[0])
        opt_blended_test = np.zeros_like(test_preds[0])
        for i in range(n_models):
            opt_blended_oof += oofs[i] * opt_mw[i]
            opt_blended_test += test_preds[i] * opt_mw[i]
            
        print("\n--- Calibrating Super Ensemble ---")
        weights_super = optimize_weights(opt_blended_oof, y_true)
        calibrated_test_2 = np.argmax(opt_blended_test * weights_super, axis=1) + 1
        
        sub2_path = "outputs/submissions/sub_final_2_super_ensemble.csv"
        sub2_df = pd.DataFrame({'id': test_ids, 'target': calibrated_test_2})
        sub2_df.to_csv(sub2_path, index=False)
        print(f"  -> Saved Submission 2: {sub2_path}")
        validate_submission(sub2_df)
        
    print("\n" + "=" * 70)
    print("ENSEMBLE PIPELINE COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    build_ensemble()
