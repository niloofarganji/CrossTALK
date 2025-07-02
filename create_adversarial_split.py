import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- Configuration ---
TRAIN_DATA_PATH = 'data/crosstalk_train (2).parquet'
TEST_DATA_PATH = 'data/crosstalk_test_inputs.parquet'
OUTPUT_PATH = 'data/crosstalk_train_with_adv_scores.parquet'
CHUNK_SIZE = 50000

# --- Helper Functions ---
def parse_fingerprints_to_sparse(fp_series):
    """Efficiently converts fingerprint strings to a SciPy sparse matrix."""
    rows, cols, data = [], [], []
    # Find the length of the fingerprint from the first valid entry
    fp_len = -1
    for fp_str in fp_series:
        if isinstance(fp_str, str) and ',' in fp_str:
            fp_len = len(fp_str.split(','))
            break
    if fp_len == -1: return csr_matrix((len(fp_series), 0))

    for i, fp_str in enumerate(tqdm(fp_series, desc=f"Parsing {fp_series.name}")):
        if isinstance(fp_str, str) and ',' in fp_str:
            bits = [int(b) for b in fp_str.split(',')]
            if len(bits) == fp_len:
                non_zero_indices = np.where(np.array(bits) == 1)[0]
                rows.extend([i] * len(non_zero_indices))
                cols.extend(non_zero_indices)
                data.extend([1] * len(non_zero_indices))
    return csr_matrix((data, (rows, cols)), shape=(len(fp_series), fp_len))

def _get_features(df, fp_cols, num_cols):
    """Parses and combines all specified features into a single sparse matrix."""
    print("Parsing fingerprint columns...")
    fp_matrices = [parse_fingerprints_to_sparse(df[fp_col]) for fp_col in fp_cols]
    
    print("Scaling numeric features...")
    scaler = StandardScaler()
    num_features_scaled = scaler.fit_transform(df[num_cols])
    
    print("Combining all features...")
    all_features = hstack(fp_matrices + [csr_matrix(num_features_scaled)], format='csr')
    return all_features

def run_adversarial_split_creation():
    """
    Trains a model to distinguish between train and test sets, then scores
    the training set examples based on how "test-like" they are.
    """
    print("--- Loading Data ---")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    # Store original train indices to map scores back later
    original_train_indices = train_df.index
    
    # Create the adversarial validation label (0 for train, 1 for test)
    train_df['is_test'] = 0
    test_df['is_test'] = 1

    # Standardize column names for consistency
    if 'AlogP' in test_df.columns:
        test_df.rename(columns={'AlogP': 'ALOGP'}, inplace=True)
    if 'AlogP' in train_df.columns:
        train_df.rename(columns={'AlogP': 'ALOGP'}, inplace=True)

    # Identify all feature columns
    fingerprint_cols = [col for col in train_df.columns if 'FP' in col or col in ['MACCS', 'RDK', 'AVALON', 'TOPTOR', 'ATOMPAIR']]
    numeric_cols = ['MW', 'ALOGP']
    
    print(f"Found {len(fingerprint_cols)} fingerprint columns and {len(numeric_cols)} numeric columns.")

    # Combine data for training the adversarial model
    combined_df = pd.concat([train_df[fingerprint_cols + numeric_cols + ['is_test']], 
                               test_df[fingerprint_cols + numeric_cols + ['is_test']]], 
                              ignore_index=True)

    X_adv = _get_features(combined_df, fingerprint_cols, numeric_cols)
    y_adv = combined_df['is_test'].values

    print("\n--- Training Adversarial 'Spy' Model ---")
    adv_model = lgb.LGBMClassifier(objective='binary', random_state=42)
    adv_model.fit(X_adv, y_adv)
    
    print("\n--- Generating Adversarial Scores ---")
    # Predict the probability of being in the test set for all data
    adv_preds = adv_model.predict_proba(X_adv)[:, 1]

    # Extract the scores that correspond to the original training data
    train_scores = adv_preds[:len(train_df)]

    # Add scores to the training dataframe that is already in memory
    print("Adding scores to the in-memory training data...")
    train_df['adversarial_score'] = train_scores

    # Remove the temporary column that was used for training the spy model
    train_df.drop(columns=['is_test'], inplace=True)

    print(f"\n--- Saving Data with Adversarial Scores ---")
    train_df.to_parquet(OUTPUT_PATH)
    print(f"Successfully saved enhanced training data to: {OUTPUT_PATH}")
    print("You can now use this file to create a more robust validation set in your main script.")


if __name__ == "__main__":
    run_adversarial_split_creation() 