import pandas as pd
import joblib
import os
import json
import numpy as np
import scipy
from scipy.sparse import hstack, csr_matrix
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

# --- Configuration ---
# List of expert models to use for the ensemble.
EXPERT_MODEL_DIRS = [
    'Exports/Expert_Models/withadvsplit/Xgboost_prob_paramtuned_ECFP6_advsplit_20250702-114351',
    'Exports/Expert_Models/withadvsplit/Xgboost_prob_paramtuned_FCFP6_advsplit_20250702-120010',
    #'Exports/Expert_Models/withadvsplit/Xgboost_prob_paramtuned_ATOMPAIR_advsplit_20250702-123546',
    'Exports/Expert_Models/withadvsplit/Xgboost_prob_paramtuned_RDK_advsplit_20250702-121358'
]

# Path to the labeled training data, used to train the meta-model.
TRAIN_DATA_PATH = 'data/crosstalk_train_with_adv_scores.parquet'

# Path to the new, unseen data you want to make predictions on.
NEW_DATA_PATH = 'data/crosstalk_test_inputs.parquet'

# Path where the final ensembled predictions will be saved.
OUTPUT_PATH = 'Predictions/ensemble_ranking_3feat.csv'
CHUNK_SIZE = 50000

# --- Helper Functions ---

def parse_fingerprints_to_sparse(fp_series):
    """Efficiently converts fingerprint strings to a SciPy sparse matrix."""
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
    """Parses and combines fingerprint columns into a single sparse matrix."""
    if not fingerprint_cols: return None
    fp_matrices = [parse_fingerprints_to_sparse(df[fp_col]) for fp_col in fingerprint_cols if fp_col in df.columns]
    return hstack(fp_matrices, format='csr') if fp_matrices else None

# --- Inference Logic ---

def run_single_model_inference(model_dir, data_path, chunk_size, eval_indices=None):
    """
    Runs inference for a single expert model. If eval_indices is provided, it runs
    on a specific subset of the data. Otherwise, it runs on the entire file.
    Returns a pandas Series of prediction probabilities, indexed by original row index.
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    config_path = os.path.join(model_dir, 'config.json')
    if not all(os.path.exists(p) for p in [model_path, config_path]):
        print(f"Warning: Model or config not found for '{model_dir}'. Skipping.")
        return None

    with open(config_path, 'r') as f: config = json.load(f)
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib')) if os.path.exists(os.path.join(model_dir, 'scaler.joblib')) else None
    svd = joblib.load(os.path.join(model_dir, 'svd_model.joblib')) if os.path.exists(os.path.join(model_dir, 'svd_model.joblib')) else None
    
    fp_features = config.get('FINGERPRINT_FEATURES', [])
    num_features = config.get('NUMERIC_FEATURES', [])
    use_svd = config.get('DIMENSIONALITY_REDUCTION', {}).get('enabled', False)
    
    all_probabilities = {}
    pf = pq.ParquetFile(data_path)
    
    eval_indices_set = set(eval_indices) if eval_indices is not None else None
    
    row_offset = 0
    desc = f"Predicting with {os.path.basename(model_dir)}"
    for batch in tqdm(pf.iter_batches(batch_size=chunk_size), total=int(np.ceil(pf.metadata.num_rows / chunk_size)), desc=desc):
        
        current_indices = range(row_offset, row_offset + len(batch))
        
        if eval_indices_set:
            target_relative_indices = [i for i, idx in enumerate(current_indices) if idx in eval_indices_set]
            if not target_relative_indices:
                row_offset += len(batch)
                continue
            chunk_df = batch.to_pandas().iloc[target_relative_indices]
            current_absolute_indices = [idx for i, idx in enumerate(current_indices) if i in target_relative_indices]
        else:
            chunk_df = batch.to_pandas()
            current_absolute_indices = current_indices

        # Standardize column names for consistency
        if 'AlogP' in chunk_df.columns:
            chunk_df = chunk_df.rename(columns={'AlogP': 'ALOGP'})

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
            for idx, prob in zip(current_absolute_indices, probabilities):
                all_probabilities[idx] = prob
        
        row_offset += len(batch)
            
    return pd.Series(all_probabilities)

def run_uncertainty_ranking_prediction():
    """
    Main function to train a stacking model and generate final predictions
    using ensemble disagreement uncertainty for enhanced ranking.
    """
    # --- 1. Train the Stacking Meta-Model ---
    print("--- Phase 1: Training Stacking Meta-Model ---")
    print(f"Loading training data from: {TRAIN_DATA_PATH}")

    all_train_data = pd.read_parquet(TRAIN_DATA_PATH)
    y_full = all_train_data['DELLabel']
    groups_full = all_train_data['DEL_ID']
    
    test_size = 0.2
    n_splits = int(np.ceil(1.0 / test_size))
    sgkf_test_split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    _, val_idx = next(sgkf_test_split.split(np.zeros(len(y_full)), y_full, groups_full))
    
    y_meta_train = y_full.iloc[val_idx]
    print(f"Created validation set of size {len(y_meta_train)} to train the meta-model.")

    meta_train_preds = {}
    for model_dir in EXPERT_MODEL_DIRS:
        model_name = os.path.basename(model_dir)
        expert_probs = run_single_model_inference(model_dir, TRAIN_DATA_PATH, CHUNK_SIZE, eval_indices=val_idx)
        if expert_probs is not None: meta_train_preds[model_name] = expert_probs

    X_meta_train = pd.DataFrame(meta_train_preds)
    X_meta_train = X_meta_train.loc[y_meta_train.index] # Align rows with labels
    
    print("Training Logistic Regression meta-model...")
    meta_model = LogisticRegression(random_state=42, class_weight='balanced')
    meta_model.fit(X_meta_train, y_meta_train)
    print("Meta-model training complete.")

    # --- 2. Generate Predictions on New, Unseen Data with Uncertainty ---
    print("\n--- Phase 2: Generating Predictions with Uncertainty Ranking ---")
    
    pf = pq.ParquetFile(NEW_DATA_PATH)
    id_col_name = 'RandomID' if 'RandomID' in pf.schema.names else 'SMILES' if 'SMILES' in pf.schema.names else None
    all_identifiers = []
    if id_col_name:
        for batch in pf.iter_batches(columns=[id_col_name]): all_identifiers.extend(batch.to_pandas()[id_col_name])
    
    # Get individual expert predictions for uncertainty calculation
    expert_predictions = {}
    for model_dir in EXPERT_MODEL_DIRS:
        model_name = os.path.basename(model_dir)
        if model_name in X_meta_train.columns:
            print(f"Getting predictions from {model_name} for uncertainty calculation...")
            expert_probs = run_single_model_inference(model_dir, NEW_DATA_PATH, CHUNK_SIZE, eval_indices=None)
            if expert_probs is not None: 
                expert_predictions[model_name] = expert_probs

    # Combine expert predictions into DataFrame
    X_meta_test = pd.DataFrame(expert_predictions)
    X_meta_test = X_meta_test[X_meta_train.columns] # Ensure column order is the same

    print("Applying trained meta-model to new data...")
    ensemble_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # --- 3. Calculate Ensemble Disagreement Uncertainty ---
    print("Calculating ensemble disagreement uncertainty...")
    
    # Calculate uncertainty as standard deviation across expert predictions
    expert_pred_array = X_meta_test.values  # Convert to numpy array for calculation
    uncertainty_scores = np.std(expert_pred_array, axis=1)  # Std dev across models (axis=1)
    
    print(f"Uncertainty statistics:")
    print(f"  Mean uncertainty: {np.mean(uncertainty_scores):.4f}")
    print(f"  Std uncertainty: {np.std(uncertainty_scores):.4f}")
    print(f"  Min uncertainty: {np.min(uncertainty_scores):.4f}")
    print(f"  Max uncertainty: {np.max(uncertainty_scores):.4f}")
    
    # --- 4. Enhanced Ranking with Uncertainty ---
    print("Applying uncertainty-enhanced ranking...")
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'index': range(len(ensemble_probs)),
        'meta_probability': ensemble_probs,
        'uncertainty': uncertainty_scores
    })
    
    # Calculate confidence score: high probability * low uncertainty
    analysis_df['confidence_score'] = analysis_df['meta_probability'] * (1 - analysis_df['uncertainty'])
    
    # Sort by confidence score (highest first)
    analysis_df = analysis_df.sort_values('confidence_score', ascending=False)
    
    print(f"Top 10 molecules by confidence score:")
    top_10 = analysis_df.head(10)
    for i, row in top_10.iterrows():
        print(f"  Rank {len(analysis_df) - list(analysis_df.index).index(i)}: "
              f"Prob={row['meta_probability']:.3f}, "
              f"Uncertainty={row['uncertainty']:.3f}, "
              f"Confidence={row['confidence_score']:.3f}")
    
    # --- 5. Create Final Submission with Uncertainty-Based Selection ---
    # Take top entries by confidence score but submit with original meta-model probabilities
    top_indices = analysis_df['index'].values
    selected_probs = analysis_df['meta_probability'].values
    
    output_df = pd.DataFrame({'DELLabel': selected_probs})
    if id_col_name and len(all_identifiers) == len(output_df):
        # Reorder identifiers according to uncertainty ranking
        reordered_identifiers = [all_identifiers[idx] for idx in top_indices]
        output_df[id_col_name] = reordered_identifiers
        output_df = output_df[[id_col_name, 'DELLabel']]
        
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    output_df.to_csv(OUTPUT_PATH, index=False)
    
    # Also save detailed analysis for inspection
    analysis_output_path = OUTPUT_PATH.replace('.csv', '_analysis.csv')
    detailed_analysis = analysis_df.copy()
    if id_col_name and len(all_identifiers) == len(detailed_analysis):
        detailed_analysis[id_col_name] = [all_identifiers[idx] for idx in detailed_analysis['index']]
        detailed_analysis = detailed_analysis[[id_col_name, 'meta_probability', 'uncertainty', 'confidence_score']]
    detailed_analysis.to_csv(analysis_output_path, index=False)
    
    print(f"\n--- Uncertainty-Enhanced Ranking Complete ---")
    print(f"Final submission saved to: {OUTPUT_PATH}")
    print(f"Detailed analysis saved to: {analysis_output_path}")
    print(f"Ranking strategy: Confidence Score = Probability × (1 - Uncertainty)")

if __name__ == "__main__":
    run_uncertainty_ranking_prediction() 