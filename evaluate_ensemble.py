import pandas as pd
import joblib
import os
import json
import numpy as np
import scipy
from scipy.sparse import hstack, csr_matrix
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# --- Configuration ---
# TODO: Before running, update this list with the correct paths to your trained expert models.
EXPERT_MODEL_DIRS = [
    'Exports/Expert_Models/Xgboost_prob_paramtuned_ECFP6_20250625-135259',
    'Exports/Expert_Models/Xgboost_prob_paramtuned_FCFP6_20250625-135920',
    'Exports/Expert_Models/Xgboost_prob_paramtuned_ATOMPAIR_20250625-135551',
]

EVAL_DATA_PATH = 'data/crosstalk_train (2).parquet'
CHUNK_SIZE = 50000

# --- Helper Functions (copied from ensemble_predict.py for consistency) ---

def parse_fingerprints_to_sparse(fp_series):
    rows, cols, data = [], [], []
    fp_len = -1
    for i, fp_str in enumerate(fp_series):
        if isinstance(fp_str, str):
            bits = [int(b) for b in fp_str.split(',')]
            if fp_len == -1: fp_len = len(bits)
            if len(bits) != fp_len: continue
            non_zero_indices = np.where(np.array(bits) == 1)[0]
            rows.extend([i] * len(non_zero_indices))
            cols.extend(non_zero_indices)
            data.extend([1] * len(non_zero_indices))
    if fp_len == -1: return csr_matrix((len(fp_series), 0))
    return csr_matrix((data, (rows, cols)), shape=(len(fp_series), fp_len))

def _parse_fingerprints_for_prediction(df, fingerprint_cols):
    if not fingerprint_cols: return None
    fp_matrices = [parse_fingerprints_to_sparse(df[fp_col]) for fp_col in fingerprint_cols if fp_col in df.columns]
    return hstack(fp_matrices, format='csr') if fp_matrices else None

def run_single_model_inference(model_dir, data_path, chunk_size, eval_indices):
    """
    Runs inference for a single expert model on a specific subset of the data.
    Returns an array of prediction probabilities for the evaluation set.
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    config_path = os.path.join(model_dir, 'config.json')
    if not all(os.path.exists(p) for p in [model_path, config_path]): return None
    
    with open(config_path, 'r') as f: config = json.load(f)
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib')) if os.path.exists(os.path.join(model_dir, 'scaler.joblib')) else None
    svd = joblib.load(os.path.join(model_dir, 'svd_model.joblib')) if os.path.exists(os.path.join(model_dir, 'svd_model.joblib')) else None
    
    fp_features = config.get('FINGERPRINT_FEATURES', [])
    num_features = config.get('NUMERIC_FEATURES', [])
    use_svd = config.get('DIMENSIONALITY_REDUCTION', {}).get('enabled', False)
    
    all_probabilities = []
    pf = pq.ParquetFile(data_path)
    
    # Convert eval_indices to a set for efficient lookup
    eval_indices_set = set(eval_indices)
    
    row_offset = 0
    for batch in tqdm(pf.iter_batches(batch_size=chunk_size), total=int(np.ceil(pf.metadata.num_rows / chunk_size)), desc=f"Predicting with {os.path.basename(model_dir)}"):
        # Identify which rows in the full dataset this batch corresponds to
        batch_indices = range(row_offset, row_offset + len(batch))
        
        # Find the intersection of this batch's indices and the desired evaluation indices
        target_indices_in_batch = [i for i, idx in enumerate(batch_indices) if idx in eval_indices_set]

        if not target_indices_in_batch:
            row_offset += len(batch)
            continue
            
        # Convert to pandas and select only the rows we need for this evaluation
        chunk_df = batch.to_pandas().iloc[target_indices_in_batch]
        
        X_fp = _parse_fingerprints_for_prediction(chunk_df, fp_features)
        
        if X_fp is not None and 'FP_N_FEATURES' in config:
            expected_fp_cols = config['FP_N_FEATURES']
            if X_fp.shape[1] < expected_fp_cols:
                padding = csr_matrix((X_fp.shape[0], expected_fp_cols - X_fp.shape[1]), dtype=X_fp.dtype)
                X_fp = hstack([X_fp, padding], format='csr')
            elif X_fp.shape[1] > expected_fp_cols: X_fp = X_fp[:, :expected_fp_cols]

        if use_svd and svd: X_fp = svd.transform(X_fp)

        X_num_scaled = scaler.transform(chunk_df[num_features]) if num_features and scaler else None

        if X_fp is not None and X_num_scaled is not None:
            X_final = hstack([X_fp, X_num_scaled], format='csr') if not use_svd else np.hstack([X_fp, X_num_scaled])
        else:
            X_final = X_fp if X_fp is not None else X_num_scaled

        if X_final is not None:
            probabilities = model.predict_proba(X_final)[:, 1]
            all_probabilities.extend(probabilities)

        row_offset += len(batch)
            
    return np.array(all_probabilities)

def evaluate_ensemble(decision_threshold):
    """Main function to evaluate the ensemble against a held-out portion of the training data."""
    print(f"--- Starting Ensemble Evaluation ---")
    print(f"Loading full training dataset from: {EVAL_DATA_PATH} to create a validation split.")
    print(f"Decision threshold for Precision/Recall/F1: {decision_threshold}")

    # 1. Load the full dataset and create the same train/test split used for the expert models.
    print("Loading full dataset and recreating the 80/20 split...")
    all_columns = pd.read_parquet(EVAL_DATA_PATH)
    y_true_full = all_columns['DELLabel']
    groups_full = all_columns['DEL_ID']

    # Use the same splitter configuration as in train.py to get the identical test set
    test_size = 0.2
    n_splits = int(np.ceil(1.0 / test_size))
    sgkf_test_split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # We only need the second split, which is the test set indices
    _, test_idx = next(sgkf_test_split.split(np.zeros(len(y_true_full)), y_true_full, groups_full))

    # Create the true validation set
    y_true_eval = y_true_full.iloc[test_idx]
    
    print(f"Created a validation set of size {len(y_true_eval)} for fair evaluation.")

    # 2. Get probability predictions from each expert model, running on the same test set portion
    all_expert_preds = {}
    for model_dir in EXPERT_MODEL_DIRS:
        model_name = os.path.basename(model_dir)
        # Note: We pass the indices to the inference function now
        expert_probs = run_single_model_inference(model_dir, EVAL_DATA_PATH, CHUNK_SIZE, test_idx)
        if expert_probs is not None and len(expert_probs) == len(y_true_eval):
            all_expert_preds[model_name] = expert_probs
        else:
            print(f"Warning: Skipping {model_name} due to error or length mismatch.")

    if not all_expert_preds:
        print("Error: No valid predictions were generated. Aborting.")
        return

    # 3. Calculate the ensembled predictions
    ensemble_probs = np.mean(list(all_expert_preds.values()), axis=0)
    all_expert_preds['Ensemble (Average)'] = ensemble_probs

    # 4. Calculate and report metrics for all models
    print("\n--- Model Performance Metrics ---")
    results = []
    for name, probs in all_expert_preds.items():
        # Convert probabilities to binary labels for classification metrics
        y_pred = (probs >= decision_threshold).astype(int)
        
        auc = roc_auc_score(y_true_eval, probs)
        precision = precision_score(y_true_eval, y_pred)
        recall = recall_score(y_true_eval, y_pred)
        f1 = f1_score(y_true_eval, y_pred)
        results.append({'Model': name, 'AUC': auc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of models on labeled training data.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold to convert probabilities to binary labels for Precision, Recall, and F1-score."
    )
    args = parser.parse_args()
    evaluate_ensemble(args.threshold) 