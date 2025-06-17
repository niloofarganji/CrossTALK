import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json
import time

from .dataset import basic_dataloader
from . import eval

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
    X, y = basic_dataloader(
        filepath=config['DATA_PATH'],
        x_cols=config['FEATURES'],
        y_col=config['LABEL'],
        max_to_load=config.get('MAX_ROWS')
    )
    print(f"Data loaded. Feature shape: {X.shape}, Label shape: {y.shape}")

    # 2. Split Data
    print("\n[2/4] Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['TEST_SIZE'], random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # 3. Train Model
    print("\n[3/4] Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate Model and Save Results
    eval.evaluate_and_save_results(model, X_val, y_val, output_dir)

    # Save the trained model artifact
    if config.get('MODEL_OUTPUT_PATH'):
        model_path = os.path.join(output_dir, config['MODEL_OUTPUT_PATH'])
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    print("\n--- Experiment Finished ---")
    return model 