import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from utils import seed_everything

def build_specialist_features(df_train, df_test):
    """
    Extract high-resolution features tailored specifically for distinguishing Class 5 vs Class 6.
    """
    print("  -> Engineering Specialist High-Resolution Features for Class 5 vs Class 6...")
    
    # 1. Base numerical features
    base_cols = [c for c in df_train.columns if c.startswith('f') and c[1:].isdigit()]
    base_cols = sorted(base_cols, key=lambda x: int(x[1:]))
    
    # Compute Cohen's d separation on train subset where target in [5, 6]
    sub_df = df_train[df_train['target'].isin([5, 6])].copy()
    c5 = sub_df[sub_df['target'] == 5]
    c6 = sub_df[sub_df['target'] == 6]
    
    diff_scores = []
    for col in base_cols:
        m5, m6 = c5[col].mean(), c6[col].mean()
        s5, s6 = c5[col].std(), c6[col].std()
        pooled_s = np.sqrt((s5**2 + s6**2)/2.0) + 1e-9
        d = abs(m6 - m5) / pooled_s
        diff_scores.append((col, d))
        
    diff_scores.sort(key=lambda x: x[1], reverse=True)
    top_cols = [x[0] for x in diff_scores[:25]]
    print(f"  -> Top 5 discriminating base features: {top_cols[:5]}")
    
    # Generate pairwise difference and ratio features for top 12
    interaction_dict_tr = {}
    interaction_dict_te = {}
    
    top_12 = top_cols[:12]
    for i in range(len(top_12)):
        for j in range(i+1, len(top_12)):
            c1, c2 = top_12[i], top_12[j]
            
            interaction_dict_tr[f'{c1}_minus_{c2}'] = df_train[c1].values - df_train[c2].values
            interaction_dict_tr[f'{c1}_plus_{c2}'] = df_train[c1].values + df_train[c2].values
            interaction_dict_tr[f'{c1}_ratio_{c2}'] = (df_train[c1].values + 1e-5) / (df_train[c2].values + 1e-5)
            
            interaction_dict_te[f'{c1}_minus_{c2}'] = df_test[c1].values - df_test[c2].values
            interaction_dict_te[f'{c1}_plus_{c2}'] = df_test[c1].values + df_test[c2].values
            interaction_dict_te[f'{c1}_ratio_{c2}'] = (df_test[c1].values + 1e-5) / (df_test[c2].values + 1e-5)
            
    df_inter_tr = pd.DataFrame(interaction_dict_tr, index=df_train.index).astype(np.float32)
    df_inter_te = pd.DataFrame(interaction_dict_te, index=df_test.index).astype(np.float32)
    
    # 2. Aggregations over top 25 features
    top_vals_tr = df_train[top_cols].values
    top_vals_te = df_test[top_cols].values
    
    agg_dict_tr = {
        'top_mean': np.nanmean(top_vals_tr, axis=1).astype(np.float32),
        'top_std': np.nanstd(top_vals_tr, axis=1).astype(np.float32),
        'top_max': np.nanmax(top_vals_tr, axis=1).astype(np.float32),
        'top_min': np.nanmin(top_vals_tr, axis=1).astype(np.float32),
        'top_sum': np.nansum(top_vals_tr, axis=1).astype(np.float32),
        'top_median': np.nanmedian(top_vals_tr, axis=1).astype(np.float32),
    }
    agg_dict_te = {
        'top_mean': np.nanmean(top_vals_te, axis=1).astype(np.float32),
        'top_std': np.nanstd(top_vals_te, axis=1).astype(np.float32),
        'top_max': np.nanmax(top_vals_te, axis=1).astype(np.float32),
        'top_min': np.nanmin(top_vals_te, axis=1).astype(np.float32),
        'top_sum': np.nansum(top_vals_te, axis=1).astype(np.float32),
        'top_median': np.nanmedian(top_vals_te, axis=1).astype(np.float32),
    }
    df_agg_tr = pd.DataFrame(agg_dict_tr, index=df_train.index)
    df_agg_te = pd.DataFrame(agg_dict_te, index=df_test.index)
    
    # 3. High-resolution SVD components (12 components)
    print("  -> Computing 12 SVD components specifically for 5 vs 6...")
    X_raw_tr = df_train[base_cols].values.astype(np.float32)
    X_raw_te = df_test[base_cols].values.astype(np.float32)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    svd = TruncatedSVD(n_components=12, random_state=42)
    
    # Fit SVD strictly on train subset target in [5, 6] to capture maximum variance within boundary
    mask_56 = df_train['target'].isin([5, 6]).values
    X_sub_imp = imputer.fit_transform(X_raw_tr[mask_56])
    X_sub_scl = scaler.fit_transform(X_sub_imp)
    svd.fit(X_sub_scl)
    
    svd_tr = svd.transform(scaler.transform(imputer.transform(X_raw_tr))).astype(np.float32)
    svd_te = svd.transform(scaler.transform(imputer.transform(X_raw_te))).astype(np.float32)
    
    svd_cols = [f'spec_svd_{i}' for i in range(12)]
    df_svd_tr = pd.DataFrame(svd_tr, columns=svd_cols, index=df_train.index)
    df_svd_te = pd.DataFrame(svd_te, columns=svd_cols, index=df_test.index)
    
    # Drop non-feature columns
    existing_feats_tr = [c for c in df_train.columns if c not in ['id', 'target', 'fold']]
    existing_feats_te = [c for c in df_test.columns if c not in ['id', 'target', 'fold']]
    
    X_train_full = pd.concat([df_train[existing_feats_tr], df_inter_tr, df_agg_tr, df_svd_tr], axis=1)
    X_test_full = pd.concat([df_test[existing_feats_te], df_inter_te, df_agg_te, df_svd_te], axis=1)
    
    print(f"  -> Total Specialist Feature Space: {X_train_full.shape[1]} features.")
    return X_train_full, X_test_full

def train_class56_specialist():
    print("=" * 70)
    print("   TRAINING 5-FOLD CLASS 5 VS CLASS 6 BINARY SPECIALIST CLASSIFIER")
    print("=" * 70)
    
    seed_everything(42)
    
    # Load feature datasets
    train_df = pd.read_feather("data/processed/train_features.feather")
    test_df = pd.read_feather("data/processed/test_features.feather")
    
    # Extract Specialist Features
    X_train_df, X_test_df = build_specialist_features(train_df, test_df)
    
    feature_cols = X_train_df.columns.tolist()
    X_test = X_test_df.values.astype(np.float32)
    
    # Filter train data for target in [5, 6]
    mask_56 = train_df['target'].isin([5, 6])
    sub_train = train_df[mask_56].copy()
    y_binary = (sub_train['target'] == 6).astype(int).values
    
    folds_sub = sub_train['fold'].values
    folds_all = train_df['fold'].values
    
    oof_binary_spec = np.zeros(len(train_df), dtype=np.float32)
    test_binary_spec = np.zeros(len(test_df), dtype=np.float32)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 25,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'n_estimators': 2000,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    aucs, accs, loglosses = [], [], []
    
    print("\nTraining 5-Fold LightGBM Binary Specialist...")
    for fold in range(5):
        trn_mask_sub = (folds_sub != fold)
        val_mask_sub = (folds_sub == fold)
        val_mask_all = (folds_all == fold)
        
        X_trn = X_train_df.loc[mask_56].loc[trn_mask_sub, feature_cols].values.astype(np.float32)
        y_trn = y_binary[trn_mask_sub]
        
        X_val = X_train_df.loc[mask_56].loc[val_mask_sub, feature_cols].values.astype(np.float32)
        y_val = y_binary[val_mask_sub]
        
        X_val_all = X_train_df.loc[val_mask_all, feature_cols].values.astype(np.float32)
        
        model = lgb.LGBMClassifier(**lgb_params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=250)
        ]
        
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        
        val_preds_sub = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds_sub)
        acc = accuracy_score(y_val, (val_preds_sub >= 0.5).astype(int))
        ll = log_loss(y_val, val_preds_sub)
        
        aucs.append(auc)
        accs.append(acc)
        loglosses.append(ll)
        print(f"  Fold {fold+1} Best Iter: {model.best_iteration_} | AUC: {auc:.6f} | Acc: {acc:.6f} | LogLoss: {ll:.6f}")
        
        # Predict on all OOF fold samples & test set
        oof_binary_spec[val_mask_all] = model.predict_proba(X_val_all)[:, 1]
        test_binary_spec += model.predict_proba(X_test)[:, 1] / 5.0
        
        del X_trn, y_trn, X_val, y_val, model
        gc.collect()
        
    print("\n" + "=" * 70)
    print(f"SPECIALIST CV RESULTS (Class 5 vs 6 Boundary):")
    print(f"  -> Mean 5-Fold ROC AUC: {np.mean(aucs):.6f} (+/- {np.std(aucs):.6f})")
    print(f"  -> Mean 5-Fold Accuracy: {np.mean(accs):.6f} (+/- {np.std(accs):.6f})")
    print(f"  -> Mean 5-Fold LogLoss:  {np.mean(loglosses):.6f} (+/- {np.std(loglosses):.6f})")
    print("=" * 70)
    
    # Save outputs as requested
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/oof_class56_specialist.npy", oof_binary_spec)
    np.save("outputs/test_class56_specialist.npy", test_binary_spec)
    
    # Also save inside outputs/oof/ and outputs/test_preds/ for convenience
    os.makedirs("outputs/oof", exist_ok=True)
    os.makedirs("outputs/test_preds", exist_ok=True)
    np.save("outputs/oof/oof_class56_specialist.npy", oof_binary_spec)
    np.save("outputs/test_preds/test_class56_specialist.npy", test_binary_spec)
    
    print("\nSaved binary OOF to outputs/oof_class56_specialist.npy")
    print("Saved binary Test to outputs/test_class56_specialist.npy")

if __name__ == "__main__":
    train_class56_specialist()
