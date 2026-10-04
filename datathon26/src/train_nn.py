import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import seed_everything, compute_macro_f1, validate_submission

class FastSimpleImputer:
    """
    Fast median imputer strictly implementing SimpleImputer(strategy='median')
    using vectorised numpy operations.
    """
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        self.statistics_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X_out = X.copy()
        nan_indices = np.where(np.isnan(X_out))
        X_out[nan_indices] = np.take(self.statistics_, nan_indices[1])
        return X_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class TabularNN(nn.Module):
    """
    3-layer MLP / ResNet architecture:
    Linear(input_dim -> 256) -> BatchNorm1d -> SiLU -> Dropout(0.25)
    -> Linear(256 -> 128) -> BatchNorm1d -> SiLU -> Dropout(0.20)
    -> Linear(128 -> 7)
    """
    def __init__(self, input_dim, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def optimize_weights(oof_probs, y_true):
    """
    Find optimal class probability weights w_1..w_7 to maximize Macro F1 using Nelder-Mead.
    y_pred = argmax(oof_probs * w) + 1
    """
    n_classes = oof_probs.shape[1]
    
    def loss_func(weights):
        scaled_probs = oof_probs * weights
        preds = np.argmax(scaled_probs, axis=1) + 1
        return -f1_score(y_true, preds, average='macro', zero_division=0)
    
    init_weights = np.ones(n_classes)
    
    print("\n--- Optimizing Class Probability Weights via Nelder-Mead ---", flush=True)
    res = minimize(
        loss_func,
        init_weights,
        method='Nelder-Mead',
        options={'maxiter': 500, 'disp': True}
    )
    
    best_weights = res.x
    best_weights = best_weights / np.max(best_weights) # normalize
    return best_weights

def run_nn_cv():
    print("=" * 70, flush=True)
    print("      5-FOLD STRATIFIED CV PYTORCH TABULAR NEURAL NETWORK", flush=True)
    print("=" * 70, flush=True)
    
    seed_everything(42)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device} | CPU threads: {torch.get_num_threads()}", flush=True)
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/oof", exist_ok=True)
    os.makedirs("outputs/test_preds", exist_ok=True)
    os.makedirs("outputs/submissions", exist_ok=True)
    
    # 1. Load Data
    print("\n[1/6] Loading Data & Stratified Folds...", flush=True)
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    folds_path = "outputs/folds.csv" if os.path.exists("outputs/folds.csv") else "data/folds.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    folds_df = pd.read_csv(folds_path)
    
    train_df['fold'] = folds_df['fold'].values
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'fold']]
    print(f"  -> Features count: {len(features)}", flush=True)
    print(f"  -> Train shape: {train_df.shape}", flush=True)
    print(f"  -> Test shape:  {test_df.shape}", flush=True)
    
    y_true_orig = train_df['target'].values
    y_true_0idx = y_true_orig - 1 # map 1..7 to 0..6
    n_classes = 7
    
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    
    batch_size_train = 2048
    batch_size_eval = 4096
    epochs = 25
    
    print("\n[2/6] Starting 5-Fold Stratified Training...", flush=True)
    fold_uncal_f1_scores = []
    
    for fold in range(5):
        t0_fold = time.time()
        print(f"\n" + "-" * 50, flush=True)
        print(f">>> FOLD {fold + 1} / 5", flush=True)
        print("-" * 50, flush=True)
        
        trn_mask = (train_df['fold'] != fold).values
        val_mask = (train_df['fold'] == fold).values
        
        X_trn_raw = train_df.loc[trn_mask, features].values.astype(np.float32)
        y_trn = y_true_0idx[trn_mask]
        
        X_val_raw = train_df.loc[val_mask, features].values.astype(np.float32)
        y_val = y_true_0idx[val_mask]
        
        X_test_raw = test_df[features].values.astype(np.float32)
        
        # 1. Median Imputation strictly inside CV fold
        imputer = FastSimpleImputer(strategy='median')
        X_trn_imp = imputer.fit_transform(X_trn_raw)
        X_val_imp = imputer.transform(X_val_raw)
        X_test_imp = imputer.transform(X_test_raw)
        
        # 2. StandardScaler strictly inside CV fold
        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(X_trn_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Balanced class weights for CrossEntropyLoss
        classes_present, counts = np.unique(y_trn, return_counts=True)
        class_weights = len(y_trn) / (n_classes * counts.astype(np.float32))
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # DataLoaders
        trn_ds = TensorDataset(torch.tensor(X_trn_scaled, dtype=torch.float32), torch.tensor(y_trn, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
        
        trn_loader = DataLoader(trn_ds, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size_eval, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size_eval, shuffle=False)
        
        # Model, Optimizer, Scheduler
        model = TabularNN(input_dim=len(features), num_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_val_f1 = 0.0
        best_val_probs = None
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in trn_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)
            
            scheduler.step()
            train_loss /= len(trn_ds)
            
            # Validation
            model.eval()
            val_probs_list = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    out = model(bx)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    val_probs_list.append(probs)
            
            val_probs_epoch = np.vstack(val_probs_list)
            val_preds_cls = np.argmax(val_probs_epoch, axis=1) + 1
            epoch_f1 = f1_score(y_true_orig[val_mask], val_preds_cls, average='macro', zero_division=0)
            
            if epoch_f1 > best_val_f1 or epoch == epochs:
                best_val_f1 = epoch_f1
                best_val_probs = val_probs_epoch
            
            if epoch % 5 == 0 or epoch == epochs:
                lr_curr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Macro F1: {epoch_f1:.6f} (Best: {best_val_f1:.6f}) | LR: {lr_curr:.6f}", flush=True)
        
        oof_probs[val_mask] = best_val_probs
        fold_uncal_f1_scores.append(best_val_f1)
        print(f"  Fold {fold + 1} Best Val Macro F1: {best_val_f1:.6f} | Time: {time.time() - t0_fold:.1f}s", flush=True)
        
        # Test predictions for this fold
        model.eval()
        test_probs_fold = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                out = model(bx)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                test_probs_fold.append(probs)
        
        test_probs_fold = np.vstack(test_probs_fold)
        test_probs += test_probs_fold / 5.0
        
        del model, optimizer, scheduler, trn_loader, val_loader, test_loader
        del X_trn_raw, X_trn_imp, X_trn_scaled, X_val_raw, X_val_imp, X_val_scaled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n" + "=" * 70, flush=True)
    print(f"Mean 5-Fold Uncalibrated Macro F1: {np.mean(fold_uncal_f1_scores):.6f} (+/- {np.std(fold_uncal_f1_scores):.6f})", flush=True)
    print("=" * 70, flush=True)
    
    # Save probability files
    print("\n[3/6] Saving OOF and Test Probabilities...", flush=True)
    np.save("outputs/oof_nn.npy", oof_probs)
    np.save("outputs/test_nn.npy", test_probs)
    np.save("outputs/oof/oof_nn.npy", oof_probs)
    np.save("outputs/test_preds/preds_nn.npy", test_probs)
    print(f"  -> Saved outputs/oof_nn.npy {oof_probs.shape}", flush=True)
    print(f"  -> Saved outputs/test_nn.npy {test_probs.shape}", flush=True)
    
    # 4. Uncalibrated OOF Macro F1
    print("\n[4/6] Evaluating Uncalibrated Full OOF Predictions...", flush=True)
    uncal_oof_preds = np.argmax(oof_probs, axis=1) + 1
    uncal_macro_f1 = f1_score(y_true_orig, uncal_oof_preds, average='macro', zero_division=0)
    print(f"==================================================", flush=True)
    print(f"   UNCALIBRATED OOF MACRO F1 SCORE: {uncal_macro_f1:.6f}", flush=True)
    print(f"==================================================", flush=True)
    
    # 5. Threshold Calibration
    print("\n[5/6] Calibrating Thresholds (Class Multipliers)...", flush=True)
    best_weights = optimize_weights(oof_probs, y_true_orig)
    
    cal_oof_preds = np.argmax(oof_probs * best_weights, axis=1) + 1
    cal_macro_f1 = f1_score(y_true_orig, cal_oof_preds, average='macro', zero_division=0)
    
    print(f"\n==================================================", flush=True)
    print(f"   UNCALIBRATED OOF MACRO F1: {uncal_macro_f1:.6f}", flush=True)
    print(f"   CALIBRATED OOF MACRO F1:   {cal_macro_f1:.6f}  (Delta: {cal_macro_f1 - uncal_macro_f1:+.6f})", flush=True)
    print(f"==================================================", flush=True)
    print(f"Optimal Class Multipliers: {np.round(best_weights, 4)}", flush=True)
    
    print("\nDetailed Calibrated Per-Class Classification Report:", flush=True)
    print(classification_report(y_true_orig, cal_oof_preds, digits=4, zero_division=0), flush=True)
    
    # 6. Save Calibrated Submission
    print("\n[6/6] Generating Calibrated Submission File...", flush=True)
    cal_test_probs = test_probs * best_weights
    cal_test_preds = np.argmax(cal_test_probs, axis=1) + 1
    
    sample_sub = pd.read_csv("data/sample_submission.csv")
    sub_df = pd.DataFrame({
        'id': sample_sub['id'].values,
        'target': cal_test_preds
    })
    
    out_sub_path = "outputs/submissions/sub_nn_calibrated.csv"
    sub_df.to_csv(out_sub_path, index=False)
    print(f"  -> Saved {out_sub_path}", flush=True)
    
    validate_submission(sub_df, sample_sub_path="data/sample_submission.csv")
    print("\n" + "=" * 70, flush=True)
    print("PYTORCH TABULAR NEURAL NETWORK PIPELINE COMPLETE", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_nn_cv()
