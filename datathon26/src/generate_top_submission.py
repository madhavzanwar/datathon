import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from optimize_thresholds import optimize_weights
from utils import validate_submission

def generate_top_submission():
    print("=" * 70)
    print("      GENERATING GRAND SUPER ENSEMBLE SUBMISSION (0.820773 CV)")
    print("=" * 70)
    
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    y_true = train_df['target'].values
    
    oof_lgb = np.load('outputs/oof/oof_lgb_fe.npy')
    oof_xgb = np.load('outputs/oof/oof_xgb_fe.npy')
    oof_cat = np.load('outputs/oof_cat.npy')
    oof_nn = np.load('outputs/oof_nn.npy')
    
    test_lgb = np.load('outputs/test_preds/preds_lgb_fe.npy')
    test_xgb = np.load('outputs/test_preds/preds_xgb_fe.npy')
    test_cat = np.load('outputs/test_cat.npy')
    test_nn = np.load('outputs/test_nn.npy')
    
    # 0.40 * NN + 0.30 * XGB + 0.20 * LGB + 0.10 * Cat
    oof_blend = 0.40 * oof_nn + 0.30 * oof_xgb + 0.20 * oof_lgb + 0.10 * oof_cat
    test_blend = 0.40 * test_nn + 0.30 * test_xgb + 0.20 * test_lgb + 0.10 * test_cat
    
    weights = optimize_weights(oof_blend, y_true)
    
    oof_preds = np.argmax(oof_blend * weights, axis=1) + 1
    f1 = f1_score(y_true, oof_preds, average='macro')
    print(f"\n Verified CV Macro F1 Score: {f1:.6f}")
    
    test_preds = np.argmax(test_blend * weights, axis=1) + 1
    
    os.makedirs("outputs/submissions", exist_ok=True)
    sub_path = "outputs/submissions/sub_grand_super_ensemble_0.820773.csv"
    sub_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSaved grand ensemble submission to: {sub_path}")
    validate_submission(sub_df)
    
if __name__ == "__main__":
    generate_top_submission()
