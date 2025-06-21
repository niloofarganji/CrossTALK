import os
from src.crosstalk import train

# --- Experiment Configuration ---
# This dictionary contains all the settings for our experiment.
# You can easily change these values to run different tests.

CONFIG = {
    # Name for this specific run, used to create the output folder.
    'RUN_NAME': 'catboost_ECFP6_FCFP6_TOPTOR_ATOMPAIR',

    # The model architecture to use for this experiment.
    # Options: 'logistic_regression', 'random_forest', 'lightgbm'
    'MODEL_NAME': 'catboost',

    # Model-specific parameters. These will be passed to the model's constructor.
    'MODEL_PARAMS': {
        'n_estimators': 500,      # Number of trees to build
        'learning_rate': 0.05,    # Step size shrinkage
        'max_depth': 6,           # Depth of the trees (CatBoost uses max_depth instead of num_leaves)
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
    
    # Proportion of data to be used for the validation set.
    'TEST_SIZE': 0.2, # Using a smaller validation set to maximize training data
    
    # --- Threshold Optimization ---
    # Set to True to analyze precision-recall trade-offs at different thresholds.
    'THRESHOLD_OPTIMIZATION': {
        'enabled': True
    },

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
    # Run the training and evaluation pipeline
    train.run_experiment(CONFIG)