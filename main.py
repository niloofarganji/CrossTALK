import os
from src.crosstalk import train

# --- Experiment Configuration ---
# This dictionary contains all the settings for our experiment.
# You can easily change these values to run different tests.

CONFIG = {
    # Name for this specific run, used to create the output folder.
    'RUN_NAME': 'baseline_logistic_regression',

    # Base directory where all experiment results will be saved.
    'EXPORT_BASE_DIR': 'Exports',
    
    # Path to the training data.
    'DATA_PATH': os.path.join('data', 'crosstalk_train (2).parquet'),
    
    # List of fingerprint columns to use as features.
    'FEATURES': ['ECFP4'],
    
    # Name of the column containing the labels.
    'LABEL': 'DELLabel',
    
    # Proportion of data to be used for the validation set.
    'TEST_SIZE': 0.2,
    
    # Number of rows to load for a quick test. Set to None to load all data.
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
