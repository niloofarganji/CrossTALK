import pandas as pd
import joblib
import os
import json
import numpy as np
import scipy
from scipy.sparse import hstack, csr_matrix
import pyarrow.parquet as pq
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Update these paths before running the script.

# Path to the saved model artifact from a previous training run.
# This should be the .joblib file containing the wrapped pipeline.
MODEL_PATH = 'Exports/xgboost2_prob_paramtuned_ECFP6_ATOMPAIR_non_numeric_20250624-220441/model.joblib'

# Path to the new, unseen data you want to make predictions on.
# This data must contain the same feature columns used for training.
NEW_DATA_PATH = 'data/crosstalk_test_inputs.parquet' 

# Path where the final output file with predictions will be saved (folder called Predictions)
OUTPUT_PATH = 'Predictions/results3.csv' 

# --- Helper Functions ---

def parse_fingerprints_to_sparse(fp_series):
    """
    Efficiently converts a pandas Series of comma-separated fingerprint strings
    into a SciPy CSR sparse matrix.
    """
    rows, cols, data = [], [], []
    fp_len = -1

    for i, fp_str in enumerate(fp_series):
        if isinstance(fp_str, str):
            bits = [int(b) for b in fp_str.split(',')]
            if fp_len == -1:
                fp_len = len(bits)
            
            if len(bits) != fp_len:
                continue 

            non_zero_indices = np.where(np.array(bits) == 1)[0]
            rows.extend([i] * len(non_zero_indices))
            cols.extend(non_zero_indices)
            data.extend([1] * len(non_zero_indices))

    if fp_len == -1:
        return csr_matrix((len(fp_series), 0))
    
    return csr_matrix((data, (rows, cols)), shape=(len(fp_series), fp_len))


def _parse_fingerprints_for_prediction(df, fingerprint_cols):
    """
    Parses string-based fingerprint columns from a DataFrame into a sparse matrix.
    """
    if not fingerprint_cols:
        return None
    
    fp_matrices = []
    for fp_col in fingerprint_cols:
        if fp_col in df.columns:
            # Efficiently convert to sparse without intermediate dense matrix
            fp_sparse_chunk = parse_fingerprints_to_sparse(df[fp_col])
            fp_matrices.append(fp_sparse_chunk)
            
    return hstack(fp_matrices, format='csr') if fp_matrices else None

# --- Main Inference Logic ---

def run_inference(model_path, data_path, output_path, chunk_size=50000):
    """
    Loads a trained model and its associated preprocessing artifacts,
    applies the same transformations to new data, and makes predictions.
    Processes the new data in chunks to handle large files.
    """
    print(f"--- Starting Inference ---")

    # 1. Validate inputs and derive paths
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return

    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.json')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    svd_path = os.path.join(model_dir, 'svd_model.joblib')

    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {model_dir}")
        print("Please ensure the model directory contains the config file from the training run.")
        return

    # 2. Load all artifacts from training
    print("Loading model and preprocessing artifacts...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = joblib.load(model_path)
    
    # Load scaler if it was used
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded scaler.")
    
    # Load SVD model if it was used
    svd = None
    if os.path.exists(svd_path):
        svd = joblib.load(svd_path)
        print("Loaded SVD model.")
        
    # Get feature names from config
    fp_features = config.get('FINGERPRINT_FEATURES', [])
    num_features = config.get('NUMERIC_FEATURES', [])
    use_svd = config.get('DIMENSIONALITY_REDUCTION', {}).get('enabled', False)

    # 3. Load and preprocess the new data in chunks
    print(f"Loading and preprocessing new data in chunks from: {data_path}")
    
    pf = pq.ParquetFile(data_path)
    
    all_probabilities = []
    all_identifiers = []
    id_col_name = None

    print(f"Processing {pf.metadata.num_rows} rows in chunks of {chunk_size}...")
    for batch in tqdm(pf.iter_batches(batch_size=chunk_size), total=int(np.ceil(pf.metadata.num_rows / chunk_size))):
        new_data_df = batch.to_pandas()

        # a) Process fingerprints
        X_fp = _parse_fingerprints_for_prediction(new_data_df, fp_features)
        if X_fp is None and not num_features:
            print("Warning: No fingerprint or numeric features found in this chunk. Skipping.")
            continue

        # b) Adjust fingerprint matrix shape to match training data.
        if X_fp is not None and 'FP_N_FEATURES' in config:
            expected_fp_cols = config['FP_N_FEATURES']
            current_fp_cols = X_fp.shape[1]
            
            if current_fp_cols != expected_fp_cols:
                if current_fp_cols < expected_fp_cols:
                    padding = scipy.sparse.csr_matrix((X_fp.shape[0], expected_fp_cols - current_fp_cols), dtype=X_fp.dtype)
                    X_fp = hstack([X_fp, padding], format='csr')
                else:
                    X_fp = X_fp[:, :expected_fp_cols]

        # c) Apply SVD if used
        if use_svd and svd is not None and X_fp is not None:
            X_fp = svd.transform(X_fp)

        # d) Process numeric features
        X_num_scaled = None
        if num_features:
            if scaler:
                X_num = new_data_df[num_features].values
                X_num_scaled = scaler.transform(X_num)
            else:
                print("Warning: Numeric features specified but no scaler found. Skipping scaling.")

        # e) Combine features
        if X_num_scaled is not None and X_fp is not None:
            if use_svd: # SVD output is dense
                X_final = np.hstack([X_fp, X_num_scaled])
            else: # Fingerprints are sparse
                X_final = hstack([X_fp, X_num_scaled], format='csr')
        elif X_fp is not None:
            X_final = X_fp
        else: # Only numeric features
            X_final = X_num_scaled
        
        # 4. Make predictions
        probabilities = model.predict_proba(X_final)[:, 1]
        all_probabilities.extend(probabilities)
        
        # 5. Collect identifiers
        if id_col_name is None:
            if 'RandomID' in new_data_df.columns: id_col_name = 'RandomID'
            elif 'SMILES' in new_data_df.columns: id_col_name = 'SMILES'

        if id_col_name:
            all_identifiers.extend(new_data_df[id_col_name].values)

    print("All chunks processed. Assembling final results...")

    # 6. Save results
    if id_col_name:
        output_df = pd.DataFrame({
            id_col_name: all_identifiers,
            'DELLabel': all_probabilities
        })
    else:
        # Fallback if no identifier column is found
        print("Warning: No 'RandomID' or 'SMILES' column found. Saving probabilities only.")
        output_df = pd.DataFrame({'DELLabel': all_probabilities})

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_df.to_csv(output_path, index=False)
    print(f"\n--- Inference Complete ---")
    print(f"Results with predictions saved to: {output_path}")
    print(f"Output file contains {len(output_df.columns)} columns and {len(output_df)} rows.")


if __name__ == "__main__":
    run_inference(MODEL_PATH, NEW_DATA_PATH, OUTPUT_PATH) 