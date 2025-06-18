import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import numpy as np

from .dataset import basic_dataloader
from . import eval
from . import models

def run_experiment(config):
    """
    Runs a full training and evaluation experiment based on a configuration.

    Args:
        config (dict): A dictionary containing the experiment configuration.
    """
    # --- Setup ---
    # Create a unique directory for this experiment's results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = config.get('RUN_NAME', 'experiment')
    output_dir = os.path.join(config['EXPORT_BASE_DIR'], f"{run_name}_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"--- Starting Experiment: {run_name} ---")
    print(f"Results will be saved to: {output_dir}")

    # Save the configuration for this run
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 1. Load Data
    print("\n[1/4] Loading data...")
    X_fp, X_num, y = basic_dataloader(
        filepath=config['DATA_PATH'],
        fingerprint_cols=config['FINGERPRINT_FEATURES'],
        numeric_cols=config.get('NUMERIC_FEATURES'),
        y_col=config['LABEL'],
        max_to_load=config.get('MAX_ROWS')
    )
    print(f"Data loaded. Fingerprint shape: {X_fp.shape if X_fp is not None else 'N/A'}, Numeric shape: {X_num.shape if X_num is not None else 'N/A'}")

    # 2. Split Data
    print("\n[2/4] Splitting data into training and validation sets using Grouped Stratification...")
    
    # Load group IDs for a leak-free split, ensuring all rows for a given compound are in the same set
    print("Loading group IDs for splitting...")
    groups = pd.read_parquet(
        config['DATA_PATH'], 
        columns=['DEL_ID']
    )['DEL_ID'].values
    if config.get('MAX_ROWS'):
        groups = groups[:config.get('MAX_ROWS')]

    # Use StratifiedGroupKFold to ensure groups (DEL_IDs) are not split across train/val
    # and that the label distribution is maintained.
    n_splits = int(1.0 / config['TEST_SIZE'])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # We only need the first split from the generator
    train_idx, val_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    
    y_train, y_val = y[train_idx], y[val_idx]
    
    X_fp_train, X_fp_val = None, None
    if X_fp is not None:
        X_fp_train, X_fp_val = X_fp[train_idx], X_fp[val_idx]
        
    X_num_train, X_num_val = None, None
    if X_num is not None:
        X_num_train, X_num_val = X_num[train_idx], X_num[val_idx]


    # 3. Scale Numeric Features and Combine
    print("\n[3/4] Scaling numeric features and combining with fingerprints...")
    X_train_final = X_fp_train
    X_val_final = X_fp_val

    if X_num is not None:
        scaler = StandardScaler()
        X_num_train_scaled = scaler.fit_transform(X_num_train)
        X_num_val_scaled = scaler.transform(X_num_val)
        
        # Combine scaled numeric features with sparse fingerprint features
        X_train_final = hstack([X_fp_train, X_num_train_scaled], format='csr')
        X_val_final = hstack([X_fp_val, X_num_val_scaled], format='csr')

    print(f"Final training feature shape: {X_train_final.shape}")
    print(f"Final validation feature shape: {X_val_final.shape}")


    # 4. Train Model
    print(f"\n[4/4] Training {config['MODEL_NAME']} model...")
    model = models.get_model(
        config['MODEL_NAME'], 
        config.get('MODEL_PARAMS')
    )
    model.fit(X_train_final, y_train)
    print("Model training complete.")

    # 5. Evaluate Model and Save Results
    eval.evaluate_and_save_results(model, X_val_final, y_val, output_dir)

    # Save the trained model artifact
    if config.get('MODEL_OUTPUT_PATH'):
        model_path = os.path.join(output_dir, config['MODEL_OUTPUT_PATH'])
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    print("\n--- Experiment Finished ---")
    return model 