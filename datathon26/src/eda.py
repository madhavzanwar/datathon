import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import seed_everything

def run_eda():
    print("=" * 70)
    print("           DATATHON 26-27: DEEP EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    seed_everything(42)
    
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_sub_path = "data/sample_submission.csv"
    
    print("\n[1/6] Ingesting Datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_sub_path)
    
    print(f"  -> Train shape: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"  -> Test shape:  {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
    print(f"  -> Sample sub:  {sample_df.shape[0]:,} rows x {sample_df.shape[1]} columns")
    
    train_mem = train_df.memory_usage(deep=True).sum() / (1024**2)
    test_mem = test_df.memory_usage(deep=True).sum() / (1024**2)
    print(f"  -> Memory Usage: Train={train_mem:.1f} MB, Test={test_mem:.1f} MB")
    
    # Check Columns
    features = [c for c in train_df.columns if c not in ['id', 'target']]
    test_features = [c for c in test_df.columns if c != 'id']
    print(f"\n[2/6] Feature Alignment Check...")
    print(f"  -> Train features count: {len(features)}")
    print(f"  -> Test features count:  {len(test_features)}")
    assert features == test_features, "Feature names mismatch between train and test!"
    print("  -> Features match exactly between train and test: f1 ... f174")
    
    # ID check
    print(f"\n[3/6] Identifier (ID) Diagnostics...")
    train_ids = set(train_df['id'])
    test_ids = set(test_df['id'])
    overlap = train_ids.intersection(test_ids)
    print(f"  -> Train ID range: [{train_df['id'].min():,}, {train_df['id'].max():,}]")
    print(f"  -> Test ID range:  [{test_df['id'].min():,}, {test_df['id'].max():,}]")
    print(f"  -> Train/Test ID Overlap: {len(overlap)} (Expected: 0)")
    assert len(overlap) == 0, "Warning: Overlapping IDs detected!"
    
    # Target Distribution
    print(f"\n[4/6] Target Class Distribution (Train)...")
    target_counts = train_df['target'].value_counts().sort_index()
    target_pcts = train_df['target'].value_counts(normalize=True).sort_index() * 100
    for cls, count in target_counts.items():
        print(f"  Class {cls}: {count:8,d} ({target_pcts[cls]:6.2f}%)")
    
    min_class = target_counts.min()
    max_class = target_counts.max()
    imbalance_ratio = max_class / min_class
    print(f"  -> Imbalance Ratio (Max / Min): {imbalance_ratio:.2f}x")
    
    # Missing Value Analysis
    print(f"\n[5/6] Missing Value Analysis...")
    train_nulls = train_df[features].isna().sum()
    test_nulls = test_df[features].isna().sum()
    
    total_train_cells = train_df.shape[0] * len(features)
    total_test_cells = test_df.shape[0] * len(features)
    train_missing_rate = (train_nulls.sum() / total_train_cells) * 100
    test_missing_rate = (test_nulls.sum() / total_test_cells) * 100
    
    print(f"  -> Total Train Missingness: {train_missing_rate:.2f}% ({train_nulls.sum():,} missing values)")
    print(f"  -> Total Test Missingness:  {test_missing_rate:.2f}% ({test_nulls.sum():,} missing values)")
    
    cols_with_missing_train = (train_nulls > 0).sum()
    cols_with_missing_test = (test_nulls > 0).sum()
    print(f"  -> Columns with missing in Train: {cols_with_missing_train} / {len(features)}")
    print(f"  -> Columns with missing in Test:  {cols_with_missing_test} / {len(features)}")
    
    train_row_nulls = train_df[features].isna().sum(axis=1)
    test_row_nulls = test_df[features].isna().sum(axis=1)
    print(f"  -> Train Row Missing Count: min={train_row_nulls.min()}, mean={train_row_nulls.mean():.2f}, max={train_row_nulls.max()}")
    print(f"  -> Test Row Missing Count:  min={test_row_nulls.min()}, mean={test_row_nulls.mean():.2f}, max={test_row_nulls.max()}")
    
    # Zero-variance & Constant Check
    std_train = train_df[features].std()
    zero_var_cols = std_train[std_train == 0].index.tolist()
    print(f"  -> Zero-variance / constant features: {len(zero_var_cols)}")
    if zero_var_cols:
        print(f"     Columns: {zero_var_cols}")
        
    # Generate 5-Fold Stratified Splits
    print(f"\n[6/6] Generating Leak-Free 5-Fold Stratified Split...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        train_df.loc[val_idx, 'fold'] = fold
        
    fold_dist = pd.crosstab(train_df['fold'], train_df['target'], normalize='index') * 100
    print("  -> Fold vs Target Class Distribution (%):")
    print(fold_dist.round(2))
    
    os.makedirs("data", exist_ok=True)
    train_df[['id', 'target', 'fold']].to_csv("data/folds.csv", index=False)
    print("  -> Saved fold mapping to data/folds.csv successfully!")
    print("=" * 70)
    print("EDA & CV HARNESS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()
