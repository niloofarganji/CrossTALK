import os
from src.crosstalk import train

# --- Experiment Configuration ---
# This dictionary contains all the settings for our experiment.
# You can easily change these values to run different tests.

CONFIG = {
    # Name for this specific run, used to create the output folder.
    'RUN_NAME': 'catboost_paramtuned_ECFP6_ATOMPAIR_non_numeric',

    # The model architecture to use for this experiment.
    # Options: 'logistic_regression', 'random_forest', 'lightgbm', 'xgboost'
    'MODEL_NAME': 'catboost',

    # Model-specific parameters for a baseline run (when tuning is disabled).
    'MODEL_PARAMS': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 5,
    },

    # --- Hyperparameter Tuning Configuration ---
    'HYPERPARAM_TUNING': {
        'enabled': True,  # Set to True to run the tuning pipeline
        'n_trials': 50,  # Number of trials for Optuna to run
        'optuna_search_space': {
            'xgboost': {
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'max_depth': ['int', 3, 10],
                'n_estimators': ['int', 200, 1000],
                'l2_leaf_reg': ['float', 1, 10, 'log'], # Alias for reg_lambda
            },
            'catboost': {
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'depth': ['int', 4, 10],
                'l2_leaf_reg': ['float', 1, 10, 'log'],
                'random_strength': ['float', 1e-9, 10, 'log'],
            },
            'lightgbm': {
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'num_leaves': ['int', 20, 150],
                'n_estimators': ['int', 200, 1000],
                'reg_alpha': ['float', 1e-8, 10.0, 'log'], # L1 regularization
                'reg_lambda': ['float', 1e-8, 10.0, 'log'], # L2 regularization
            },
            'random_forest': {
                'n_estimators': ['int', 100, 1000],
                'max_depth': ['int', 10, 50],
                'min_samples_leaf': ['int', 1, 10]
            }
        }
    },

    # --- Threshold Optimization ---
    # Set to True to analyze precision-recall trade-offs at different thresholds.
    'THRESHOLD_OPTIMIZATION': {
        'enabled': False
    },

    # Base directory where all experiment results will be saved.
    'EXPORT_BASE_DIR': 'Exports',
    
    # Path to the training data.
    'DATA_PATH': os.path.join('data', 'crosstalk_train (2).parquet'),
    
    # List of fingerprint columns to use.
    'FINGERPRINT_FEATURES': ['ECFP6', 'ATOMPAIR'],

    # List of numeric columns to use.
    'NUMERIC_FEATURES': [],
    
    # Name of the column containing the labels.
    'LABEL': 'DELLabel',
    
    # Proportion of data to be used for the TEST set.
    'TEST_SIZE': 0.1, 
    
    # Proportion of data to be used for the VALIDATION set (from the initial training split)
    'VALIDATION_SIZE': 0.1,

    # Set to None to run on the entire dataset.
    'MAX_ROWS': None,
    
    # Filename for the saved model artifact within the run's output directory.
    'MODEL_OUTPUT_PATH': 'model.joblib'
}


if __name__ == "__main__":
    """
    This is the main entry point of the script.
    It triggers the experiment defined by the CONFIG dictionary.
    """
    # Adjust output directory if hyperparameter tuning is enabled
    if CONFIG.get('HYPERPARAM_TUNING', {}).get('enabled', False):
        CONFIG['EXPORT_BASE_DIR'] = os.path.join(CONFIG['EXPORT_BASE_DIR'], 'Hyperparameter_Tuning')

    # Run the training and evaluation pipeline
    train.run_experiment(CONFIG) 