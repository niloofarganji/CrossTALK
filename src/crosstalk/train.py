#import standard libraries:
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, GroupKFold
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import numpy as np
from sklearn.decomposition import TruncatedSVD
import optuna

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

    # 1. Load Data
    print("\n[1/6] Loading data...")
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

    # Load cluster IDs if using cluster-based splitting
    cluster_ids = None
    if config.get('CLUSTER_SPLIT', {}).get('enabled', False):
        print("Loading cluster IDs for diversity-aware splitting.")
        cluster_ids = pd.read_parquet(config['DATA_PATH'], columns=['cluster_id'])['cluster_id'].values
        if config.get('MAX_ROWS'):
            cluster_ids = cluster_ids[:len(y)]

    # Store the exact number of fingerprint features from the training set.
    # This is crucial for ensuring the prediction data has the same shape.
    config['FP_N_FEATURES'] = X_fp.shape[1]
    
    print(f"--- Starting Experiment: {run_name} ---")
    print(f"Results will be saved to: {output_dir}")

    # Save the final configuration, now including the number of fingerprint features.
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 2. Split Data (Train / Validation / Test)
    if config.get('HYPERPARAM_TUNING', {}).get('enabled', False):
        print("\n[2/6] Splitting data into train/validation (for tuning) and test sets...")
    else:
        print("\n[2/6] Splitting data into train and test sets...")
    
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
    
    # 2.5. Dimensionality Reduction (Optional)
    if config.get('DIMENSIONALITY_REDUCTION', {}).get('enabled', False):
        dr_config = config['DIMENSIONALITY_REDUCTION']
        n_components = dr_config['n_components']
        print(f"\n[2.5/6] Applying TruncatedSVD to reduce fingerprint dimensions to {n_components}...")

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit on the training data and transform both train and test data
        X_fp_train_val = svd.fit_transform(X_fp_train_val)
        X_fp_test = svd.transform(X_fp_test)
        
        # Save the fitted SVD model
        svd_model_path = os.path.join(output_dir, 'svd_model.joblib')
        joblib.dump(svd, svd_model_path)
        print(f"SVD model saved to {svd_model_path}")

        # Plot the explained variance
        eval.plot_explained_variance(svd, output_dir)

    # 3. Scale numeric features and Combine
    if X_num is not None:
        print("\n[3/6] Scaling numeric features and combining with fingerprints...")
        scaler = StandardScaler()
        X_num_train_val_scaled = scaler.fit_transform(X_num_train_val)
        X_num_test_scaled = scaler.transform(X_num_test)
        
        # Save the fitted scaler
        scaler_model_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_model_path)
        print(f"Scaler model saved to {scaler_model_path}")
        
        # If we used dimensionality reduction, X_fp is dense. Use np.hstack.
        if config.get('DIMENSIONALITY_REDUCTION', {}).get('enabled', False):
            X_train_val_final = np.hstack([X_fp_train_val, X_num_train_val_scaled])
            X_test_final = np.hstack([X_fp_test, X_num_test_scaled])
        else:
            # Otherwise, X_fp is sparse. Use scipy.sparse.hstack.
            X_train_val_final = hstack([X_fp_train_val, X_num_train_val_scaled], format='csr')
            X_test_final = hstack([X_fp_test, X_num_test_scaled], format='csr')
    else:
        X_train_val_final = X_fp_train_val
        X_test_final = X_fp_test

    # 4. Hyperparameter Tuning (Optional)
    best_params = config.get('MODEL_PARAMS', {})
    if config.get('HYPERPARAM_TUNING', {}).get('enabled', False):
        print("\n[4/6] Starting hyperparameter tuning with Optuna...")
        
        hp_config = config['HYPERPARAM_TUNING']
        search_space_config = hp_config['optuna_search_space'][config['MODEL_NAME']]
        n_trials = hp_config.get('n_trials', 50)

        def objective(trial):
            params = {}
            for name, space in search_space_config.items():
                if space[0] == 'int':
                    params[name] = trial.suggest_int(name, space[1], space[2])
                elif space[0] == 'float' and len(space) == 3:
                    params[name] = trial.suggest_float(name, space[1], space[2])
                elif space[0] == 'float' and len(space) == 4 and space[3] == 'log':
                    params[name] = trial.suggest_float(name, space[1], space[2], log=True)
                elif space[0] == 'categorical':
                     params[name] = trial.suggest_categorical(name, space[1])
            
            model = models.get_model(config['MODEL_NAME'], model_params=params)
            
            # Use diversity-aware GroupKFold if enabled, otherwise use default stratified split
            if config.get('CLUSTER_SPLIT', {}).get('enabled', False):
                cv_splitter = GroupKFold(n_splits=5)
                # Note: With pure GroupKFold, we use the cluster_ids as the grouping variable
                cv_groups = cluster_ids[train_val_idx]
            else:
                cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                cv_groups = groups_train_val

            # Use cross_val_score for efficient evaluation
            f1_scores = cross_val_score(
                model, 
                X_train_val_final, 
                y_train_val, 
                groups=cv_groups,
                cv=cv_splitter, 
                scoring='f1_weighted',
                n_jobs=1
            )
            return np.mean(f1_scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        print(f"Best parameters found: {best_params}")
        
        # Save tuning results
        tuning_results_df = study.trials_dataframe()
        tuning_results_path = os.path.join(output_dir, 'tuning_results.csv')
        tuning_results_df.to_csv(tuning_results_path, index=False)
        print(f"Tuning results saved to {tuning_results_path}")

        # Save best params
        best_params_path = os.path.join(output_dir, 'best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {best_params_path}")

    # 5. Train Final Model
    print(f"\n[5/6] Training final {config['MODEL_NAME']} model...")
    final_model = models.get_model(config['MODEL_NAME'], best_params)
    final_model.fit(X_train_val_final, y_train_val)
    print("Final model training complete.")

    # 6. Evaluate Final Model on Test Set
    print("\n[6/6] Evaluating model on test set...")
    eval.evaluate_and_save_results(final_model, X_test_final, y_test, output_dir, result_name='test')

    # 7. Perform Threshold Optimization Analysis (Optional)
    optimal_threshold = 0.5 # Default threshold
    if config.get('THRESHOLD_OPTIMIZATION', {}).get('enabled', False):
        print("\n[7/7] Performing threshold optimization analysis...")
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