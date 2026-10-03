import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils import compute_macro_f1, validate_submission

def optimize_weights(oof_probs, y_true):
    """
    Find optimal class probability weights w_1..w_7 to maximize Macro F1.
    y_pred = argmax(oof_probs * w) + 1
    """
    n_classes = oof_probs.shape[1]
    
    # Negative Macro F1 objective
    def loss_func(weights):
        scaled_probs = oof_probs * weights
        preds = np.argmax(scaled_probs, axis=1) + 1
        return -compute_macro_f1(y_true, preds, verbose=False)
    
    init_weights = np.ones(n_classes)
    
    print("\n--- Optimizing Class Probability Weights via Powell / Nelder-Mead ---")
    res = minimize(
        loss_func,
        init_weights,
        method='Nelder-Mead',
        options={'maxiter': 500, 'disp': True}
    )
    
    best_weights = res.x
    best_weights = best_weights / np.max(best_weights) # normalize
    
    # Calculate baseline vs post-optimization F1
    raw_preds = np.argmax(oof_probs, axis=1) + 1
    raw_f1 = compute_macro_f1(y_true, raw_preds, verbose=False)
    
    opt_preds = np.argmax(oof_probs * best_weights, axis=1) + 1
    print(f"\n[Before Optimization] Raw Argmax Macro F1: {raw_f1:.6f}")
    print(f"[After Optimization]  Calibrated Macro F1: {res.fun * -1:.6f} (Delta: {res.fun * -1 - raw_f1:+.6f})")
    print(f"Optimal Class Multipliers: {np.round(best_weights, 4)}")
    
    compute_macro_f1(y_true, opt_preds, verbose=True)
    
    return best_weights

def apply_and_save(oof_path, test_path, out_sub_path, folds_path="data/folds.csv", sample_path="data/sample_submission.csv"):
    oof_probs = np.load(oof_path)
    test_probs = np.load(test_path)
    folds_df = pd.read_csv(folds_path)
    y_true = folds_df['target'].values
    
    weights = optimize_weights(oof_probs, y_true)
    
    # Apply to test set
    scaled_test = test_probs * weights
    final_test_preds = np.argmax(scaled_test, axis=1) + 1
    
    sample_df = pd.read_csv(sample_path)
    sub_df = pd.DataFrame({
        'id': sample_df['id'].values,
        'target': final_test_preds
    })
    sub_df.to_csv(out_sub_path, index=False)
    print(f"\nSaved calibrated submission to {out_sub_path}")
    validate_submission(sub_df, sample_sub_path=sample_path)

if __name__ == "__main__":
    apply_and_save(
        "outputs/oof/oof_lgb_baseline.npy",
        "outputs/test_preds/preds_lgb_baseline.npy",
        "outputs/submissions/sub_lgb_baseline_calibrated.csv"
    )
