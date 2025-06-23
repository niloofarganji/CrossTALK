import os
from src.crosstalk import train_pipe

# --- Experiment Configuration ---
# This dictionary contains all the settings for our experiment.
# You can easily change these values to run different tests.

CONFIG = {
    # Name for this specific run, used to create the output folder.
    'RUN_NAME': 'xgboost_pipe_tuned',

    # The model architecture to use for this experiment.
    'MODEL_NAME': 'xgboost',
    
    # --- Hyperparameter Tuning Configuration ---
    'HYPERPARAM_TUNING': {
        'enabled': True,
        'param_grids': {
            'xgboost': {
                'preprocessor__fingerprints__svd__n_components': [150, 200, 250],
                'classifier__n_estimators': [200, 500],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5, 7]
            },
            'catboost': {
                'preprocessor__fingerprints__svd__n_components': [150, 200, 250],
                'classifier__n_estimators': [200, 500],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__depth': [4, 6, 8],
                'classifier__l2_leaf_reg': [1, 3, 5]
            },
            'lightgbm': {
                'preprocessor__fingerprints__svd__n_components': [150, 200, 250],
                'classifier__n_estimators': [200, 500],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__num_leaves': [31, 40, 50]
            },
            'random_forest': {
                'preprocessor__fingerprints__svd__n_components': [150, 200, 250],
                'classifier__n_estimators': [100, 200, 500],
                'classifier__max_features': ['sqrt', 'log2']
            }
        }
    },

    # --- Threshold Optimization ---
    # Set to True to analyze precision-recall trade-offs at different thresholds.
    'THRESHOLD_OPTIMIZATION': {
        'enabled': True
    },

    # Base directory where all experiment results will be saved.
    'EXPORT_BASE_DIR': 'Exports',
    
    # Path to the training data.
    'DATA_PATH': os.path.join('data', 'crosstalk_train (2).parquet'),
    
    # List of fingerprint columns to use.
    'FINGERPRINT_FEATURES': ['ECFP6', 'FCFP6', 'TOPTOR', 'ATOMPAIR'],

    # List of numeric columns to use.
    'NUMERIC_FEATURES': ['MW', 'ALOGP'],
    
    # Name of the column containing the labels.
    'LABEL': 'DELLabel',
    
    # Proportion of data to be used for the TEST set.
    'TEST_SIZE': 0.1, 
    
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
        CONFIG['EXPORT_BASE_DIR'] = os.path.join(CONFIG['EXPORT_BASE_DIR'], 'Exports_Pipe')

    # Run the training and evaluation pipeline
    train_pipe.run_experiment_pipe(CONFIG) 