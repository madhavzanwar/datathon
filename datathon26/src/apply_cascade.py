import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, compute_macro_f1, validate_submission
from optimize_thresholds import optimize_weights

def apply_cascade_refinement(raw_probs, spec_probs, alpha=1.0):
    """
    Refine multi-class probabilities using the binary specialist predictions for Class 5 vs 6.
    
    raw_probs: shape (N, 7) - raw multi-class probabilities (0-indexed: col 4 is Class 5, col 5 is Class 6)
    spec_probs: shape (N,) - binary probability of Class 6 given target in [5, 6]
    alpha: blend parameter between raw 5-6 ratio and specialist ratio (default 1.0 for direct replacement)
    """
    refined_probs = raw_probs.copy()
    
    p5_raw = raw_probs[:, 4]
    p6_raw = raw_probs[:, 5]
    p56_sum = p5_raw + p6_raw
    
    # Raw ratio for Class 6 within the 5/6 subset
    raw_ratio_6 = np.where(p56_sum > 1e-12, p6_raw / (p56_sum + 1e-12), 0.5)
    
    # Blended ratio
    blended_ratio_6 = (1.0 - alpha) * raw_ratio_6 + alpha * spec_probs
    
    refined_probs[:, 4] = p56_sum * (1.0 - blended_ratio_6)
    refined_probs[:, 5] = p56_sum * blended_ratio_6
    
    return refined_probs

def run_cascade_pipeline():
    print("=" * 70)
    print("      MILESTONE: CASCADE REFINEMENT PIPELINE (CLASS 5 vs CLASS 6)")
    print("=" * 70)
    
    seed_everything(42)
    
    # Load ground truth folds and sample submission
    folds_df = pd.read_csv("data/folds.csv")
    y_true = folds_df['target'].values
    
    sample_sub = pd.read_csv("data/sample_submission.csv")
    test_ids = sample_sub['id'].values
    
    # Load Specialist predictions
    spec_oof_path = "outputs/oof_class56_specialist.npy"
    spec_test_path = "outputs/test_class56_specialist.npy"
    
    assert os.path.exists(spec_oof_path), f"Missing specialist OOF: {spec_oof_path}"
    assert os.path.exists(spec_test_path), f"Missing specialist Test: {spec_test_path}"
    
    spec_oof = np.load(spec_oof_path)
    spec_test = np.load(spec_test_path)
    print(f"Loaded Specialist Predictions: OOF shape {spec_oof.shape}, Test shape {spec_test.shape}")
    
    # Find all multi-class OOF arrays
    oof_files = sorted(glob.glob("outputs/oof/*.npy"))
    if not oof_files:
        oof_files = sorted(glob.glob("outputs/oof_*.npy"))
        
    print(f"\nEvaluating Cascade Refinement across {len(oof_files)} multi-class models:")
    
    best_cascade_oof = None
    best_cascade_test = None
    best_cascade_f1 = 0.0
    best_model_name = ""
    
    results = []
    
    for f in oof_files:
        base_name = os.path.basename(f).replace("oof_", "").replace(".npy", "")
        if "class56_specialist" in base_name:
            continue
            
        test_file = f"outputs/test_preds/preds_{base_name}.npy"
        if not os.path.exists(test_file):
            test_file = f"outputs/test_{base_name}.npy"
            
        oof_raw = np.load(f)
        has_test = os.path.exists(test_file)
        test_raw = np.load(test_file) if has_test else None
        
        # 1. Raw Argmax Performance
        raw_preds = np.argmax(oof_raw, axis=1) + 1
        raw_f1 = f1_score(y_true, raw_preds, average='macro', zero_division=0)
        
        # 2. Direct Cascade Replacement (alpha = 1.0)
        oof_casc_direct = apply_cascade_refinement(oof_raw, spec_oof, alpha=1.0)
        casc_direct_preds = np.argmax(oof_casc_direct, axis=1) + 1
        casc_direct_f1 = f1_score(y_true, casc_direct_preds, average='macro', zero_division=0)
        
        # 3. Optimal Alpha Search (0.0 to 1.0)
        best_alpha = 1.0
        best_alpha_f1 = casc_direct_f1
        for a in np.linspace(0.0, 1.0, 21):
            oof_c = apply_cascade_refinement(oof_raw, spec_oof, alpha=a)
            p_c = np.argmax(oof_c, axis=1) + 1
            score = f1_score(y_true, p_c, average='macro', zero_division=0)
            if score > best_alpha_f1:
                best_alpha_f1 = score
                best_alpha = a
                
        # 4. Calibrated Thresholding Post-Cascade
        weights = optimize_weights(oof_casc_direct, y_true)
        calib_preds = np.argmax(oof_casc_direct * weights, axis=1) + 1
        casc_calib_f1 = f1_score(y_true, calib_preds, average='macro', zero_division=0)
        
        print(f"\n--- Model: [{base_name}] ---")
        print(f"  Raw Argmax Macro F1:              {raw_f1:.6f}")
        print(f"  Direct Cascade (alpha=1.0) F1:   {casc_direct_f1:.6f} (Gain: {casc_direct_f1 - raw_f1:+.6f})")
        print(f"  Optimal Blended Cascade (a={best_alpha:.2f}): {best_alpha_f1:.6f} (Gain: {best_alpha_f1 - raw_f1:+.6f})")
        print(f"  Post-Cascade Calibrated F1:      {casc_calib_f1:.6f} (Gain over Raw: {casc_calib_f1 - raw_f1:+.6f})")
        
        results.append({
            'Model': base_name,
            'Raw Macro F1': raw_f1,
            'Direct Cascade F1': casc_direct_f1,
            'Direct Gain': casc_direct_f1 - raw_f1,
            'Opt Alpha': best_alpha,
            'Opt Alpha F1': best_alpha_f1,
            'Calibrated Cascade F1': casc_calib_f1,
            'Total Gain': casc_calib_f1 - raw_f1
        })
        
        if test_raw is not None and casc_calib_f1 > best_cascade_f1:
            best_cascade_f1 = casc_calib_f1
            best_model_name = base_name
            test_casc_direct = apply_cascade_refinement(test_raw, spec_test, alpha=best_alpha)
            best_cascade_test = np.argmax(test_casc_direct * weights, axis=1) + 1
            best_cascade_oof = oof_casc_direct
            
    # Summary Table
    print("\n" + "=" * 70)
    print("            CASCADE REFINEMENT SUMMARY RESULTS")
    print("=" * 70)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("=" * 70)
    
    # Save Best Refined Submission
    if best_cascade_test is not None:
        sub_path = f"outputs/submissions/sub_cascade_{best_model_name}.csv"
        sub_df = pd.DataFrame({
            'id': test_ids,
            'target': best_cascade_test
        })
        sub_df.to_csv(sub_path, index=False)
        print(f"\nSaved Best Cascade Submission ({best_model_name}): {sub_path}")
        validate_submission(sub_df)
        
    print("\n" + "=" * 70)
    print("CASCADE REFINEMENT COMPLETED")
    print("=" * 70)
    return res_df

if __name__ == "__main__":
    run_cascade_pipeline()
