import os
import gc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def extract_features(df, is_train=True, svd_model=None, imputer=None, scaler=None, n_components=8):
    """
    Extract robust row-level statistical features and dimensionality reduction embeddings
    from noisy numerical columns f1..f174.
    """
    feature_cols = [c for c in df.columns if c.startswith('f') and c[1:].isdigit()]
    feature_cols = sorted(feature_cols, key=lambda x: int(x[1:]))
    
    print(f"Extracting features from {len(feature_cols)} base features across {len(df):,} rows...")
    
    X_raw = df[feature_cols].values.astype(np.float32)
    
    # 1. Row-level Missingness Statistics
    print("  -> Computing row-level missingness and aggregation stats...")
    row_nan_count = np.isnan(X_raw).sum(axis=1).astype(np.int16)
    row_nan_ratio = (row_nan_count / len(feature_cols)).astype(np.float32)
    
    # Fast row-wise statistics ignoring NaNs
    row_mean = np.nanmean(X_raw, axis=1).astype(np.float32)
    row_std = np.nanstd(X_raw, axis=1).astype(np.float32)
    row_min = np.nanmin(X_raw, axis=1).astype(np.float32)
    row_max = np.nanmax(X_raw, axis=1).astype(np.float32)
    row_sum = np.nansum(X_raw, axis=1).astype(np.float32)
    row_median = np.nanmedian(X_raw, axis=1).astype(np.float32)
    row_q25 = np.nanpercentile(X_raw, 25, axis=1).astype(np.float32)
    row_q75 = np.nanpercentile(X_raw, 75, axis=1).astype(np.float32)
    row_iqr = (row_q75 - row_q25).astype(np.float32)
    
    # Combine stats into DataFrame
    stats_dict = {
        'row_nan_count': row_nan_count,
        'row_nan_ratio': row_nan_ratio,
        'row_mean': row_mean,
        'row_std': row_std,
        'row_min': row_min,
        'row_max': row_max,
        'row_sum': row_sum,
        'row_median': row_median,
        'row_q25': row_q25,
        'row_q75': row_q75,
        'row_iqr': row_iqr
    }
    stats_df = pd.DataFrame(stats_dict, index=df.index)
    
    # 2. SVD / PCA Latent Components (Fitted on Train only to prevent leakage)
    print("  -> Computing SVD latent representations...")
    if is_train:
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        X_imputed = imputer.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_imputed)
        svd_feats = svd_model.fit_transform(X_scaled).astype(np.float32)
    else:
        assert imputer is not None and scaler is not None and svd_model is not None, "Transformers must be provided for test data!"
        X_imputed = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imputed)
        svd_feats = svd_model.transform(X_scaled).astype(np.float32)
        
    svd_cols = [f'svd_comp_{i}' for i in range(n_components)]
    svd_df = pd.DataFrame(svd_feats, columns=svd_cols, index=df.index)
    
    # Concatenate base features + stats + SVD
    result_df = pd.concat([df[['id']], df[feature_cols], stats_df, svd_df], axis=1)
    if 'target' in df.columns:
        result_df['target'] = df['target']
    if 'fold' in df.columns:
        result_df['fold'] = df['fold']
        
    print(f"  -> Engineered dataset shape: {result_df.shape}")
    
    del X_raw, X_imputed, X_scaled, svd_feats, stats_dict, stats_df, svd_df
    gc.collect()
    
    return result_df, svd_model, imputer, scaler

if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    folds_df = pd.read_csv("data/folds.csv")
    train_df['fold'] = folds_df['fold']
    
    test_df = pd.read_csv("data/test.csv")
    
    train_fe, svd_m, imp_m, scl_m = extract_features(train_df, is_train=True)
    test_fe, _, _, _ = extract_features(test_df, is_train=False, svd_model=svd_m, imputer=imp_m, scaler=scl_m)
    
    print("Saving processed feature datasets to feather/parquet for ultra-fast loading...")
    os.makedirs("data/processed", exist_ok=True)
    train_fe.to_feather("data/processed/train_features.feather")
    test_fe.to_feather("data/processed/test_features.feather")
    print("Feature generation completed and saved to data/processed/!")
