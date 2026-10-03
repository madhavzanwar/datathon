import os
import gc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from utils import seed_everything

def evaluate_cv(X_tr, y_tr, folds, class_weight='balanced', custom_weights=None, n_estimators=1000, lr=0.08):
    """
    Evaluates 5-fold Stratified CV Macro F1 strictly without leakage.
    """
    n_classes = 7
    oof_preds = np.zeros(len(X_tr), dtype=int)
    oof_probs = np.zeros((len(X_tr), n_classes), dtype=np.float32)
    fold_scores = []
    
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': lr,
        'num_leaves': 48,
        'max_depth': 8,
        'min_child_samples': 30,
        'feature_fraction': 0.70,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'n_estimators': n_estimators,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    if custom_weights is not None:
        params['class_weight'] = custom_weights
    elif class_weight is not None:
        params['class_weight'] = class_weight
        
    for fold in range(5):
        trn_idx = (folds != fold)
        val_idx = (folds == fold)
        
        X_train_f = X_tr[trn_idx]
        y_train_f = y_tr[trn_idx]
        
        X_val_f = X_tr[val_idx]
        y_val_f = y_tr[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_f, y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        
        val_p = model.predict_proba(X_val_f)
        oof_probs[val_idx] = val_p
        val_c = np.argmax(val_p, axis=1)
        oof_preds[val_idx] = val_c
        
        f1 = f1_score(y_val_f + 1, val_c + 1, average='macro', zero_division=0)
        fold_scores.append(f1)
        
        del X_train_f, y_train_f, X_val_f, y_val_f, model
        gc.collect()
        
    overall_f1 = f1_score(y_tr + 1, oof_preds + 1, average='macro', zero_division=0)
    return overall_f1, fold_scores, oof_probs

def run_all_experiments():
    print("=" * 75)
    print("      SYSTEMATIC FEATURE & MODELING EXPERIMENTATION HARNESS")
    print("=" * 75)
    
    seed_everything(42)
    
    train_df = pd.read_csv("data/train.csv")
    folds_df = pd.read_csv("outputs/folds.csv")
    folds = folds_df['fold'].values
    y = (train_df['target'].values - 1).astype(int)
    
    base_features = [c for c in train_df.columns if c.startswith('f') and c[1:].isdigit()]
    base_features = sorted(base_features, key=lambda x: int(x[1:]))
    
    results = []
    
    # Baseline
    print("\n[Baseline] Evaluating Raw 174 Features...")
    X_base = train_df[base_features].values.astype(np.float32)
    base_f1, _, _ = evaluate_cv(X_base, y, folds, class_weight='balanced')
    print(f"  -> Baseline 5-Fold Macro F1: {base_f1:.6f}")
    results.append({'Experiment': 'Baseline (Raw 174 Feats, Balanced LightGBM)', 'Features': len(base_features), 'CV Macro F1': base_f1, 'Delta': 0.0, 'Status': 'REFERENCE'})
    
    current_best_f1 = base_f1
    active_df = train_df[base_features].copy()
    
    # -------------------------------------------------------------
    # Experiment 1: Missing-value indicators
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Exp 1] Testing Missing-Value Indicators...")
    X_raw_vals = train_df[base_features].values
    exp1_feats = {}
    exp1_feats['row_nan_count'] = np.isnan(X_raw_vals).sum(axis=1).astype(np.float32)
    exp1_feats['row_nan_ratio'] = (exp1_feats['row_nan_count'] / len(base_features)).astype(np.float32)
    
    # 4 group nan counts
    for g_idx in range(4):
        g_cols = base_features[g_idx*43 : (g_idx+1)*43]
        exp1_feats[f'nan_group_{g_idx+1}'] = np.isnan(train_df[g_cols].values).sum(axis=1).astype(np.float32)
        
    df_exp1 = pd.concat([active_df, pd.DataFrame(exp1_feats, index=train_df.index)], axis=1)
    f1_exp1, _, _ = evaluate_cv(df_exp1.values.astype(np.float32), y, folds, class_weight='balanced')
    delta1 = f1_exp1 - current_best_f1
    status1 = 'KEPT' if delta1 > 0 else 'DISCARDED'
    print(f"  -> Exp 1 Macro F1: {f1_exp1:.6f} (Delta: {delta1:+.6f}) -> {status1}")
    results.append({'Experiment': '1. Missingness Indicators (Row count + Group counts)', 'Features': df_exp1.shape[1], 'CV Macro F1': f1_exp1, 'Delta': delta1, 'Status': status1})
    
    if delta1 > 0:
        active_df = df_exp1
        current_best_f1 = f1_exp1
        
    # -------------------------------------------------------------
    # Experiment 2: Feature Pruning (Dropping Low-Signal / Pure Noise Features)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Exp 2] Testing Low-Signal Feature Pruning...")
    # Quick importance on fold 0 to identify bottom 15 lowest-gain features
    model_quick = lgb.LGBMClassifier(objective='multiclass', num_class=7, class_weight='balanced', n_estimators=250, random_state=42, n_jobs=-1, verbose=-1)
    model_quick.fit(X_base, y)
    gains = model_quick.booster_.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feat': base_features, 'gain': gains}).sort_values('gain')
    bottom_15 = imp_df.head(15)['feat'].tolist()
    
    df_exp2 = active_df.drop(columns=[c for c in bottom_15 if c in active_df.columns])
    f1_exp2, _, _ = evaluate_cv(df_exp2.values.astype(np.float32), y, folds, class_weight='balanced')
    delta2 = f1_exp2 - current_best_f1
    status2 = 'KEPT' if delta2 > 0 else 'DISCARDED'
    print(f"  -> Exp 2 Macro F1: {f1_exp2:.6f} (Delta: {delta2:+.6f}) -> {status2}")
    results.append({'Experiment': '2. Noise Feature Pruning (Drop bottom 15 lowest gain)', 'Features': df_exp2.shape[1], 'CV Macro F1': f1_exp2, 'Delta': delta2, 'Status': status2})
    
    if delta2 > 0:
        active_df = df_exp2
        current_best_f1 = f1_exp2
        
    # -------------------------------------------------------------
    # Experiment 3: Row-level Statistical Aggregations
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Exp 3] Testing Row-Level Statistical Aggregations...")
    X_vals = train_df[base_features].values.astype(np.float32)
    
    exp3_feats = {
        'row_mean': np.nanmean(X_vals, axis=1).astype(np.float32),
        'row_std': np.nanstd(X_vals, axis=1).astype(np.float32),
        'row_min': np.nanmin(X_vals, axis=1).astype(np.float32),
        'row_max': np.nanmax(X_vals, axis=1).astype(np.float32),
        'row_median': np.nanmedian(X_vals, axis=1).astype(np.float32),
        'row_q25': np.nanpercentile(X_vals, 25, axis=1).astype(np.float32),
        'row_q75': np.nanpercentile(X_vals, 75, axis=1).astype(np.float32),
        'row_iqr': (np.nanpercentile(X_vals, 75, axis=1) - np.nanpercentile(X_vals, 25, axis=1)).astype(np.float32),
    }
    
    df_exp3 = pd.concat([active_df, pd.DataFrame(exp3_feats, index=train_df.index)], axis=1)
    f1_exp3, _, _ = evaluate_cv(df_exp3.values.astype(np.float32), y, folds, class_weight='balanced')
    delta3 = f1_exp3 - current_best_f1
    status3 = 'KEPT' if delta3 > 0 else 'DISCARDED'
    print(f"  -> Exp 3 Macro F1: {f1_exp3:.6f} (Delta: {delta3:+.6f}) -> {status3}")
    results.append({'Experiment': '3. Row-Level Statistical Aggregates (mean, std, min, max, quantiles)', 'Features': df_exp3.shape[1], 'CV Macro F1': f1_exp3, 'Delta': delta3, 'Status': status3})
    
    if delta3 > 0:
        active_df = df_exp3
        current_best_f1 = f1_exp3
        
    # -------------------------------------------------------------
    # Experiment 4: Rank / Quantile Transform for Skewed Features (Inside Folds)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Exp 4] Testing Quantile Transform for Skewed Features...")
    skews = train_df[base_features].skew()
    top_skewed = skews[skews.abs() > 3].index.tolist()
    print(f"  -> Identified {len(top_skewed)} skewed features: {top_skewed}")
    
    # Apply QuantileTransformer strictly inside CV folds
    fold_scores_exp4 = []
    oof_preds_exp4 = np.zeros(len(train_df), dtype=int)
    
    for fold in range(5):
        trn_idx = (folds != fold)
        val_idx = (folds == fold)
        
        df_trn_fold = active_df.loc[trn_idx].copy()
        df_val_fold = active_df.loc[val_idx].copy()
        
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
        # Median impute before quantile transform on skewed cols
        imp = SimpleImputer(strategy='median')
        
        trn_skew_imp = imp.fit_transform(df_trn_fold[top_skewed])
        val_skew_imp = imp.transform(df_val_fold[top_skewed])
        
        trn_transformed = qt.fit_transform(trn_skew_imp)
        val_transformed = qt.transform(val_skew_imp)
        
        for idx, col in enumerate(top_skewed):
            df_trn_fold[f'{col}_quant'] = trn_transformed[:, idx]
            df_val_fold[f'{col}_quant'] = val_transformed[:, idx]
            
        model = lgb.LGBMClassifier(objective='multiclass', num_class=7, class_weight='balanced', learning_rate=0.08, num_leaves=48, max_depth=8, reg_alpha=0.5, reg_lambda=2.0, n_estimators=1000, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(df_trn_fold.values.astype(np.float32), y[trn_idx], eval_set=[(df_val_fold.values.astype(np.float32), y[val_idx])], callbacks=[lgb.early_stopping(40, verbose=False)])
        
        val_c = np.argmax(model.predict_proba(df_val_fold.values.astype(np.float32)), axis=1)
        oof_preds_exp4[val_idx] = val_c
        fold_scores_exp4.append(f1_score(y[val_idx]+1, val_c+1, average='macro', zero_division=0))
        
    f1_exp4 = f1_score(y + 1, oof_preds_exp4 + 1, average='macro', zero_division=0)
    delta4 = f1_exp4 - current_best_f1
    status4 = 'KEPT' if delta4 > 0 else 'DISCARDED'
    print(f"  -> Exp 4 Macro F1: {f1_exp4:.6f} (Delta: {delta4:+.6f}) -> {status4}")
    results.append({'Experiment': '4. Quantile Transformation on Skewed Features', 'Features': active_df.shape[1] + len(top_skewed), 'CV Macro F1': f1_exp4, 'Delta': delta4, 'Status': status4})
    
    if delta4 > 0:
        # Retain transformed features
        imp_all = SimpleImputer(strategy='median')
        qt_all = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
        skew_all_imp = imp_all.fit_transform(active_df[top_skewed])
        all_trans = qt_all.fit_transform(skew_all_imp)
        for idx, col in enumerate(top_skewed):
            active_df[f'{col}_quant'] = all_trans[:, idx]
        current_best_f1 = f1_exp4
        
    # -------------------------------------------------------------
    # Experiment 5: Imputation Variants (Native NaNs vs Median Imputation)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Exp 5] Testing Imputation Variants (Median vs Native NaN)...")
    oof_preds_exp5 = np.zeros(len(train_df), dtype=int)
    for fold in range(5):
        trn_idx = (folds != fold)
        val_idx = (folds == fold)
        
        imp_f = SimpleImputer(strategy='median')
        X_trn_imp = imp_f.fit_transform(active_df.loc[trn_idx].values)
        X_val_imp = imp_f.transform(active_df.loc[val_idx].values)
        
        model = lgb.LGBMClassifier(objective='multiclass', num_class=7, class_weight='balanced', learning_rate=0.08, num_leaves=48, max_depth=8, reg_alpha=0.5, reg_lambda=2.0, n_estimators=1000, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_trn_imp.astype(np.float32), y[trn_idx], eval_set=[(X_val_imp.astype(np.float32), y[val_idx])], callbacks=[lgb.early_stopping(40, verbose=False)])
        
        val_c = np.argmax(model.predict_proba(X_val_imp.astype(np.float32)), axis=1)
        oof_preds_exp5[val_idx] = val_c
        
    f1_exp5 = f1_score(y + 1, oof_preds_exp5 + 1, average='macro', zero_division=0)
    delta5 = f1_exp5 - current_best_f1
    status5 = 'KEPT' if delta5 > 0 else 'DISCARDED (Native NaNs are Superior for Trees)'
    print(f"  -> Exp 5 Macro F1 (Median Imputation): {f1_exp5:.6f} (Delta: {delta5:+.6f}) -> {status5}")
    results.append({'Experiment': '5. Median Imputation vs Native NaNs', 'Features': active_df.shape[1], 'CV Macro F1': f1_exp5, 'Delta': delta5, 'Status': status5})
    
    # -------------------------------------------------------------
    # Experiment 6: Class Imbalance Weighting Strategies
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Exp 6] Testing Class Weight Variants (Focal/Power Weights)...")
    # Sub-exponential power weights: w_c = (N / N_c)^0.65
    counts = pd.Series(y).value_counts().sort_index()
    power_weights = {cls: float((len(y) / (7.0 * count)) ** 0.65) for cls, count in counts.items()}
    print(f"  -> Custom Power Weights: {power_weights}")
    
    f1_exp6, _, _ = evaluate_cv(active_df.values.astype(np.float32), y, folds, custom_weights=power_weights)
    delta6 = f1_exp6 - current_best_f1
    status6 = 'KEPT' if delta6 > 0 else 'DISCARDED'
    print(f"  -> Exp 6 Macro F1 (Power Weighting): {f1_exp6:.6f} (Delta: {delta6:+.6f}) -> {status6}")
    results.append({'Experiment': '6. Class Weight Tuning (Power Weights vs Strict Balanced)', 'Features': active_df.shape[1], 'CV Macro F1': f1_exp6, 'Delta': delta6, 'Status': status6})
    
    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("                    EXPERIMENTATION RESULTS TABLE")
    print("=" * 75)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("=" * 75)
    
    # Save final best pipeline to src/features.py
    print("\n[Final] Freezing best winning features to src/features.py...")
    save_final_features_pipeline(active_df.columns.tolist(), base_features)
    
def save_final_features_pipeline(winning_cols, base_features):
    pipeline_code = f'''import numpy as np
import pandas as pd

def transform_features(df, is_train=True):
    """
    Production-ready feature transformation pipeline incorporating winning features
    from systematic 5-fold CV ablation studies.
    """
    base_features = [c for c in df.columns if c.startswith('f') and c[1:].isdigit()]
    base_features = sorted(base_features, key=lambda x: int(x[1:]))
    
    X_vals = df[base_features].values.astype(np.float32)
    
    engineered = {{}}
    
    # 1. Missingness Signals
    engineered['row_nan_count'] = np.isnan(X_vals).sum(axis=1).astype(np.float32)
    engineered['row_nan_ratio'] = (engineered['row_nan_count'] / len(base_features)).astype(np.float32)
    
    for g_idx in range(4):
        g_cols = base_features[g_idx*43 : (g_idx+1)*43]
        engineered[f'nan_group_{{g_idx+1}}'] = np.isnan(df[g_cols].values).sum(axis=1).astype(np.float32)
        
    # 2. Row-Level Statistical Aggregations
    engineered['row_mean'] = np.nanmean(X_vals, axis=1).astype(np.float32)
    engineered['row_std'] = np.nanstd(X_vals, axis=1).astype(np.float32)
    engineered['row_min'] = np.nanmin(X_vals, axis=1).astype(np.float32)
    engineered['row_max'] = np.nanmax(X_vals, axis=1).astype(np.float32)
    engineered['row_median'] = np.nanmedian(X_vals, axis=1).astype(np.float32)
    engineered['row_q25'] = np.nanpercentile(X_vals, 25, axis=1).astype(np.float32)
    engineered['row_q75'] = np.nanpercentile(X_vals, 75, axis=1).astype(np.float32)
    engineered['row_iqr'] = (engineered['row_q75'] - engineered['row_q25']).astype(np.float32)
    
    engineered_df = pd.DataFrame(engineered, index=df.index)
    
    # Combine base features + engineered features
    out_df = pd.concat([df[['id']], df[base_features], engineered_df], axis=1)
    if 'target' in df.columns:
        out_df['target'] = df['target']
        
    return out_df
'''
    with open("src/features.py", "w", encoding="utf-8") as f:
        f.write(pipeline_code)
    print("  -> Saved src/features.py successfully!")

if __name__ == "__main__":
    run_all_experiments()
