import pandas as pd
import joblib
import os
import json
import numpy as np
import scipy
from scipy.sparse import hstack

# --- Configuration ---
# IMPORTANT: Update these paths before running the script.

# Path to the saved model artifact from a previous training run.
# This should be the .joblib file containing the wrapped pipeline.
MODEL_PATH = 'Exports/xgboost_ECFP6_FCFP6_TOPTOR_ATOMPAIR_20250623-084656/model.joblib'

# Path to the new, unseen data you want to make predictions on.
# This data must contain the same feature columns used for training.
NEW_DATA_PATH = 'data/crosstalk_test_20250305_inputs.parquet' 

# Path where the final output file with predictions will be saved (folder called Predictions)
OUTPUT_PATH = 'Predictions/results1.csv' 

# --- Helper Functions ---

def _parse_fingerprints_for_prediction(df, fingerprint_cols):
    """
    Parses string-based fingerprint columns from a DataFrame into a sparse matrix.
    """
    if not fingerprint_cols:
        return None
    
    fp_matrices = []
    for fp_col in fingerprint_cols:
        if fp_col in df.columns:
            # Split comma-separated strings and convert to a numeric sparse matrix
            exploded = df[fp_col].str.split(',', expand=True).astype(np.int8)
            fp_matrices.append(scipy.sparse.csr_matrix(exploded))
            
    return hstack(fp_matrices, format='csr') if fp_matrices else None

# --- Main Inference Logic ---

def run_inference(model_path, data_path, output_path):
    """
    Loads a trained model and its associated preprocessing artifacts,
    applies the same transformations to new data, and makes predictions.
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

    # 3. Load and preprocess the new data
    print(f"Loading and preprocessing new data from: {data_path}")
    new_data_df = pd.read_parquet(data_path)

    # a) Process fingerprints
    X_fp = _parse_fingerprints_for_prediction(new_data_df, fp_features)
    if X_fp is not None:
        print(f"Initial fingerprint shape: {X_fp.shape}")

        # b) Adjust fingerprint matrix shape to match training data.
        # This MUST happen BEFORE any SVD is applied.
        if 'FP_N_FEATURES' in config:
            expected_fp_cols = config['FP_N_FEATURES']
            current_fp_cols = X_fp.shape[1]
            
            if current_fp_cols != expected_fp_cols:
                print(f"Padding/truncating fingerprint matrix from {current_fp_cols} to {expected_fp_cols} features.")
                if current_fp_cols < expected_fp_cols:
                    # Pad with zeros
                    padding = scipy.sparse.csr_matrix((X_fp.shape[0], expected_fp_cols - current_fp_cols), dtype=X_fp.dtype)
                    X_fp = hstack([X_fp, padding], format='csr')
                else:
                    # Truncate
                    X_fp = X_fp[:, :expected_fp_cols]
            print(f"Adjusted fingerprint shape: {X_fp.shape}")

    # c) Apply SVD if used
    if use_svd and svd is not None and X_fp is not None:
        print("Applying SVD transformation...")
        X_fp = svd.transform(X_fp)
        print(f"Shape after SVD: {X_fp.shape}")

    # d) Process numeric features
    X_num_scaled = None
    if num_features:
        if scaler:
            print("Scaling numeric features...")
            X_num = new_data_df[num_features].values
            X_num_scaled = scaler.transform(X_num)
            print(f"Shape of scaled numeric features: {X_num_scaled.shape}")
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
    
    print("Data preprocessed. Preparing to make predictions...")

    # 4. Make predictions
    print(f"X_final shape: {X_final.shape}")
    predictions = model.predict(X_final)
    print(f"Predictions generated for {len(predictions)} samples.")

    # 5. Save results
    output_df = new_data_df.copy()
    output_df['predicted_label'] = predictions

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_df.to_csv(output_path, index=False)
    print(f"\n--- Inference Complete ---")
    print(f"Results with predictions saved to: {output_path}")


if __name__ == "__main__":
    run_inference(MODEL_PATH, NEW_DATA_PATH, OUTPUT_PATH) 