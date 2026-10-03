import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

def seed_everything(seed=42):
    """Set random seeds across python, numpy, os for strict reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def compute_macro_f1(y_true, y_pred, verbose=True):
    """Compute Macro F1 and per-class F1 scores."""
    # Ensure targets are integers
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if verbose:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        print(f"\n--- Validation Performance ---")
        print(f"Overall Macro F1: {macro_f1:.6f}")
        print("\nPer-class Metrics:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        
    return macro_f1

def validate_submission(sub_df, sample_sub_path="data/sample_submission.csv"):
    """
    Strict integrity check for submission dataframe.
    Verifies:
    1. Exactly 2 columns: id, target
    2. Exact row count matches sample_submission.csv
    3. Exactly matches test IDs in identical sequence
    4. Target is integers 1 to 7 with zero NaNs/nulls
    """
    sample_df = pd.read_csv(sample_sub_path)
    
    assert list(sub_df.columns) == ['id', 'target'], f"Invalid columns: {list(sub_df.columns)}, expected ['id', 'target']"
    assert len(sub_df) == len(sample_df), f"Row count mismatch: got {len(sub_df)}, expected {len(sample_df)}"
    assert (sub_df['id'].values == sample_df['id'].values).all(), "ID column does not exactly match sample_submission!"
    assert sub_df['target'].isna().sum() == 0, f"Found {sub_df['target'].isna().sum()} NaN values in target!"
    assert sub_df['target'].dtype in [np.int32, np.int64, int], f"Target must be integer dtype, got {sub_df['target'].dtype}"
    
    valid_classes = set(range(1, 8))
    predicted_classes = set(sub_df['target'].unique())
    assert predicted_classes.issubset(valid_classes), f"Found invalid class labels: {predicted_classes - valid_classes}"
    
    print(f"[OK] Submission Validation Passed! {len(sub_df):,} rows, columns: {list(sub_df.columns)}")
    print("Class Distribution in Predictions:")
    print(sub_df['target'].value_counts(normalize=True).sort_index().apply(lambda x: f"{x*100:.2f}%"))
    return True
