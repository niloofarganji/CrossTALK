import os
from src.crosstalk import train

# --- Experiment Configuration ---
# This dictionary contains all the settings for our experiment.
# You can easily change these values to run different tests.

CONFIG = {
    # Name for this specific run, used to create the output folder.
    'RUN_NAME': 'xgboost_paramtuned_connectivity_fps',

    # The model architecture to use for this experiment.
    # Options: 'logistic_regression', 'random_forest', 'lightgbm', 'xgboost'
    'MODEL_NAME': 'xgboost',

    # Model-specific parameters for a baseline run (when tuning is disabled).
    'MODEL_PARAMS': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 5,
    },

    # --- Hyperparameter Tuning Configuration ---
    'HYPERPARAM_TUNING': {
        'enabled': True,  # Set to True to run the tuning pipeline
        'param_grids': {
            'xgboost': {
                'n_estimators': [200, 500],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'lightgbm': {
                'n_estimators': [200, 500],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 40, 50]
            },
            'random_forest': {
                'n_estimators': [100, 200, 500],
                'max_features': ['sqrt', 'log2'],
                'min_samples_leaf': [1, 5, 10]
            }
        }
    },

    # Base directory where all experiment results will be saved.
    'EXPORT_BASE_DIR': 'Exports',
    
    # Path to the training data.
    'DATA_PATH': os.path.join('data', 'crosstalk_train (2).parquet'),
    
    # List of fingerprint columns to use.
    'FINGERPRINT_FEATURES': ['ECFP4'],

    # List of numeric columns to use.
    'NUMERIC_FEATURES': ['MW', 'ALOGP'],
    
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
