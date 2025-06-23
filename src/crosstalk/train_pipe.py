#import standard libraries:
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
import joblib
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

#import custom modules from within the crosstalk project:
from .dataset_pipe import basic_dataloader_pipe, FingerprintTransformer
from . import eval
from . import models

def run_experiment_pipe(config):
    """
    Runs a full training and evaluation experiment using a scikit-learn Pipeline.
    """
    # --- 1. Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = config.get('RUN_NAME', 'experiment')
    output_dir = os.path.join(config['EXPORT_BASE_DIR'], f"{run_name}_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"--- Starting Pipeline Experiment: {run_name} ---")
    print(f"Results will be saved to: {output_dir}")

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # --- 2. Load Data ---
    print("\n[1/5] Loading data for pipeline...")
    fp_features = config['FINGERPRINT_FEATURES']
    num_features = config.get('NUMERIC_FEATURES', [])
    all_features = fp_features + num_features
    
    X, y = basic_dataloader_pipe(
        filepath=config['DATA_PATH'],
        all_feature_cols=all_features,
        y_col=config['LABEL'],
        max_to_load=config.get('MAX_ROWS')
    )
    groups = pd.read_parquet(config['DATA_PATH'], columns=['DEL_ID'])['DEL_ID']
    if config.get('MAX_ROWS'):
        groups = groups.head(len(y))

    # --- 3. Define Preprocessing Pipeline ---
    print("\n[2/5] Defining preprocessing and model pipeline...")

    # Create a pipeline for processing the combined fingerprint features
    fp_pipeline = Pipeline([
        ('parser', FingerprintTransformer()),
        ('svd', TruncatedSVD(random_state=42))
    ])

    # Create a preprocessor that applies different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('fingerprints', fp_pipeline, fp_features),
            ('numeric', StandardScaler(), num_features)
        ],
        remainder='drop'
    )

    # --- 4. Define the Full Model Pipeline ---
    base_model = models.get_model(config['MODEL_NAME'])
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])

    # --- 5. Split Data and Run GridSearch ---
    print("\n[3/5] Splitting data and starting hyperparameter tuning...")
    test_size = config['TEST_SIZE']
    n_splits = int(np.ceil(1.0 / test_size))
    sgkf_test_split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_val_idx, test_idx = next(sgkf_test_split.split(X, y, groups))

    X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    param_grid = config['HYPERPARAM_TUNING']['param_grids'][config['MODEL_NAME']]
    cv_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=cv_splitter,
        n_jobs=1,
        verbose=3
    )
    grid_search.fit(X_train_val, y_train_val, groups=groups_train_val)
    
    best_pipeline = grid_search.best_estimator_
    print(f"\nBest parameters found: {grid_search.best_params_}")
    
    # Save tuning results
    # ... (code to save results remains similar)

    # --- 6. Evaluate, Optimize Threshold, and Save Final Model ---
    print("\n[4/5] Evaluating final pipeline on test set...")
    eval.evaluate_and_save_results(best_pipeline, X_test, y_test, output_dir, result_name='test')
    
    optimal_threshold = 0.5
    if config.get('THRESHOLD_OPTIMIZATION', {}).get('enabled', False):
        print("\n[5/5] Performing threshold optimization analysis...")
        y_pred_proba_test = best_pipeline.predict_proba(X_test)[:, 1]
        opt_dir = os.path.join(output_dir, 'threshold_optimization')
        if not os.path.exists(opt_dir): os.makedirs(opt_dir)
        optimal_threshold = eval.analyze_thresholds(y_test, y_pred_proba_test, opt_dir)

    if config.get('MODEL_OUTPUT_PATH'):
        print("\nWrapping final pipeline with optimal threshold...")
        final_model_to_save = models.ThresholdedClassifier(model=best_pipeline, threshold=optimal_threshold)
        model_path = os.path.join(output_dir, config['MODEL_OUTPUT_PATH'])
        joblib.dump(final_model_to_save, model_path)
        print(f"Final wrapped pipeline saved to {model_path}")

    print("\n--- Pipeline Experiment Finished ---")
    return final_model_to_save

