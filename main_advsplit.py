import os
from src.crosstalk import train_advsplit

# --- Experiment Configuration ---
# This dictionary contains all the settings for our experiment.
# You can easily change these values to run different tests.

CONFIG = {
    # Name for this specific run, used to create the output folder.
    'RUN_NAME': 'Xgboost_prob_paramtuned_ATOMPAIR_advsplit',

    # The model architecture to use for this experiment.
    # Options: 'logistic_regression', 'random_forest', 'lightgbm'
    'MODEL_NAME': 'xgboost',

    # Model-specific parameters. These will be passed to the model's constructor.
    #after hyperparameter tuning
    #from Exports/Hyperparameter_Tuning/xgboost_paramtuned_ECFP6_FCFP6_TOPTOR_ATOMPAIR_20250621-143003/best_params.json
    
    #for xgboost: 
    'MODEL_PARAMS': {
        'learning_rate': 0.1,
        'max_depth': 7,
        'n_estimators': 500
    },
    
    #for catboost: 
    #'MODEL_PARAMS': {
    #    "learning_rate": 0.2141837713010419,
    #    "depth": 10,
    #    "l2_leaf_reg": 1.261357543385574,
    #    "random_strength": 0.0005881339039579889
    #}

    #'MODEL_PARAMS': {
    #    'n_estimators': 200,      # A good number of trees.
    #    'max_depth': 20,          # Limit tree depth to prevent overfitting.
    #    'min_samples_leaf': 5,    # Require at least 5 samples in a leaf node.
    #    'n_jobs': -1,             # Use all available CPU cores.
    #    'random_state': 42        # For reproducibility.
    #},

    #parameters for LogisticRegression
    #'MODEL_PARAMS': {
    #    'penalty': 'l1',
    #    'C': 1.0,
    #    'solver': 'liblinear'
    #},

    #Set Hyperparameter Tuning to False
    'HYPERPARAM_TUNING': {
        'enabled': False
    },

    # Base directory where all experiment results will be saved.
    'EXPORT_BASE_DIR': 'Exports\Expert_Models\withadvsplit',
    
    # Path to the training data with adversarial scores.
    'DATA_PATH': os.path.join('data', 'crosstalk_train_with_adv_scores.parquet'),
    
    # List of fingerprint columns to use.
    'FINGERPRINT_FEATURES': ['ATOMPAIR'],
    # List of numeric columns to use.
    
    'NUMERIC_FEATURES': ['MW', 'ALOGP'],
    
    # Name of the column containing the labels.
    'LABEL': 'DELLabel',
    
    # Proportion of data to be used for the validation set (using adversarial scores).
    'TEST_SIZE': 0.15, # Using adversarial validation with 15% test set
    
    # --- Threshold Optimization ---
    # Set to True to analyze precision-recall trade-offs at different thresholds.
    'THRESHOLD_OPTIMIZATION': {
        'enabled': False
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
    train_advsplit.run_experiment(CONFIG) 