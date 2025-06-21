#import standard libraries:
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import numpy as np

#import custom modules from within the crosstalk project:
from .dataset import basic_dataloader
from . import eval
from . import models

def run_experiment(config):
    """
    Runs a full training and evaluation experiment based on a configuration,
    with an optional hyperparameter tuning stage.
    """
    # --- Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = config.get('RUN_NAME', 'experiment')
    output_dir = os.path.join(config['EXPORT_BASE_DIR'], f"{run_name}_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"--- Starting Experiment: {run_name} ---")
    print(f"Results will be saved to: {output_dir}")

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 1. Load Data
    print("\n[1/5] Loading data...")
    X_fp, X_num, y = basic_dataloader(
        filepath=config['DATA_PATH'],
        fingerprint_cols=config['FINGERPRINT_FEATURES'],
        numeric_cols=config.get('NUMERIC_FEATURES'),
        y_col=config['LABEL'],
        max_to_load=config.get('MAX_ROWS')
    )
    groups = pd.read_parquet(config['DATA_PATH'], columns=['DEL_ID'])['DEL_ID'].values
    if config.get('MAX_ROWS'):
        groups = groups[:len(y)]

    # 2. Split Data (Train / Validation / Test)
    print("\n[2/5] Splitting data into train, validation, and test sets...")
    
    # We calculate the number of splits needed to achieve the desired test set size
    test_size = config['TEST_SIZE']
    n_splits = int(np.ceil(1.0 / test_size))

    # Split out the test set first using the group information
    sgkf_test_split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(sgkf_test_split.split(np.zeros(len(y)), y, groups))

    # Create the final test set
    y_test = y[test_idx]
    X_fp_test, X_num_test = (X_fp[test_idx], X_num[test_idx]) if X_num is not None else (X_fp[test_idx], None)

    # Create the training + validation set that will be used for tuning
    y_train_val = y[train_val_idx]
    X_fp_train_val = X_fp[train_val_idx]
    X_num_train_val = X_num[train_val_idx] if X_num is not None else None
    groups_train_val = groups[train_val_idx]
    
    # Scale numeric features - this is a critical step
    if X_num is not None:
        scaler = StandardScaler()
        X_num_train_val_scaled = scaler.fit_transform(X_num_train_val)
        X_num_test_scaled = scaler.transform(X_num_test)
        
        # Combine scaled numeric features with sparse fingerprint features for the final matrices
        X_train_val_final = hstack([X_fp_train_val, X_num_train_val_scaled], format='csr')
        X_test_final = hstack([X_fp_test, X_num_test_scaled], format='csr')
    else:
        X_train_val_final = X_fp_train_val
        X_test_final = X_fp_test

    # 3. Hyperparameter Tuning (Optional)
    best_params = config['MODEL_PARAMS']
    if config.get('HYPERPARAM_TUNING', {}).get('enabled', False):
        print("\n[3/5] Starting hyperparameter tuning with GridSearchCV...")
        
        param_grid = config['HYPERPARAM_TUNING']['param_grids'][config['MODEL_NAME']]
        base_model = models.get_model(config['MODEL_NAME'])
        
        # Use StratifiedGroupKFold for the inner cross-validation during tuning
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=cv_splitter,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train_val_final, y_train_val, groups=groups_train_val)
        
        best_params = grid_search.best_params_
        print(f"Best parameters found: {best_params}")
        
        # Save tuning results
        tuning_results_df = pd.DataFrame(grid_search.cv_results_)
        tuning_results_path = os.path.join(output_dir, 'tuning_results.csv')
        tuning_results_df.to_csv(tuning_results_path, index=False)
        print(f"Tuning results saved to {tuning_results_path}")

        # Save best params
        best_params_path = os.path.join(output_dir, 'best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {best_params_path}")

    # 4. Train Final Model
    print(f"\n[4/5] Training final {config['MODEL_NAME']} model...")
    final_model = models.get_model(config['MODEL_NAME'], best_params)
    final_model.fit(X_train_val_final, y_train_val)
    print("Final model training complete.")

    # 5. Evaluate Final Model on Test Set
    eval.evaluate_and_save_results(final_model, X_test_final, y_test, output_dir, result_name='test')

    # 6. Perform Threshold Optimization Analysis (Optional)
    optimal_threshold = 0.5 # Default threshold
    if config.get('THRESHOLD_OPTIMIZATION', {}).get('enabled', False):
        print("\n[6/6] Performing threshold optimization analysis...")
        y_pred_proba_test = final_model.predict_proba(X_test_final)[:, 1]
        
        # Create a sub-folder within the main run directory for threshold results
        run_specific_opt_dir = os.path.join(output_dir, 'threshold_optimization')
        if not os.path.exists(run_specific_opt_dir):
            os.makedirs(run_specific_opt_dir)

        optimal_threshold = eval.analyze_thresholds(y_test, y_pred_proba_test, run_specific_opt_dir)

    # Save the trained model artifact
    if config.get('MODEL_OUTPUT_PATH'):
        model_path = os.path.join(output_dir, config['MODEL_OUTPUT_PATH'])
        
        model_to_save = final_model
        # If optimization was run, wrap the model with the best threshold
        if config.get('THRESHOLD_OPTIMIZATION', {}).get('enabled', False):
            print(f"\nWrapping model with optimal threshold: {optimal_threshold:.4f}")
            model_to_save = models.ThresholdedClassifier(model=final_model, threshold=optimal_threshold)
        
        joblib.dump(model_to_save, model_path)
        print(f"Model saved to {model_path}")

    print("\n--- Experiment Finished ---")
    return final_model 