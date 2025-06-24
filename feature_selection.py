import os
import json
import pandas as pd
import importlib.util
from copy import deepcopy
from datetime import datetime
from glob import glob

# This assumes the script is run from the project root.
from src.crosstalk import train

# --- Configuration for the Feature Selection Experiment ---

# Define the combinations of fingerprint features you want to test.
# You can add or remove combinations from this list.
FEATURE_COMBINATIONS_TO_TEST = [
    ['TOPTOR', 'ECFP6'],
    ['TOPTOR', 'FCFP6'],
    ['TOPTOR', 'ATOMPAIR'],
    ['ATOMPAIR', 'ECFP6'],
    ['ATOMPAIR', 'FCFP6'],
    ['ECFP6', 'FCFP6'],
]

# Path to the main configuration file.
BASE_CONFIG_PATH = 'main.py'

# Directory to save all outputs from this experiment.
EXPERIMENT_OUTPUT_DIR = 'Exports/Feature_Selection/Feature_Selection_Analysis/feat_num'


def run_feature_selection():
    """
    Automates the process of training models with different feature combinations
    and generates a summary report of their performance.
    """
    print("--- Starting Automated Feature Selection ---")
    
    # Load the base configuration from main.py
    print(f"Loading base configuration from: {BASE_CONFIG_PATH}")
    spec = importlib.util.spec_from_file_location("config_module", BASE_CONFIG_PATH)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    base_config = config_module.CONFIG

    # Ensure the main output directory exists
    if not os.path.exists(EXPERIMENT_OUTPUT_DIR):
        os.makedirs(EXPERIMENT_OUTPUT_DIR)

    results_summary = []
    
    # --- Part 1: Run all training experiments ---
    for i, feature_set in enumerate(FEATURE_COMBINATIONS_TO_TEST):
        print(f"\n--- Running Experiment {i+1}/{len(FEATURE_COMBINATIONS_TO_TEST)} ---")
        print(f"Features: {feature_set}")
        
        # Create a deep copy to avoid modifying the original config
        current_config = deepcopy(base_config)
        
        # Modify the config for the current run
        run_name_prefix = f"featsel__{'_'.join(feature_set)}"
        current_config['FINGERPRINT_FEATURES'] = feature_set
        current_config['RUN_NAME'] = run_name_prefix
        current_config['EXPORT_BASE_DIR'] = EXPERIMENT_OUTPUT_DIR
        
        # Run the existing training pipeline with the modified config
        train.run_experiment(current_config)

    print("\n\n--- All experiments complete. Aggregating results... ---")

    # --- Part 2: Aggregate results and create a report ---
    all_run_dirs = glob(os.path.join(EXPERIMENT_OUTPUT_DIR, 'featsel_*'))

    for run_dir in all_run_dirs:
        metrics_path = os.path.join(run_dir, 'test_metrics.json')
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract the run name and features from the directory path
            run_name = os.path.basename(run_dir)
            features_str = run_name.split('__')[1].split('_20')[0] # Clean up name
            
            # Extract key performance metrics for the positive class (label '1')
            precision = metrics.get('1', {}).get('precision', 0)
            recall = metrics.get('1', {}).get('recall', 0)
            f1_score = metrics.get('1', {}).get('f1-score', 0)
            roc_auc = metrics.get('roc_auc_score', 0)

            results_summary.append({
                'features_tested': features_str.replace('_', ', '),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'roc_auc_score': roc_auc
            })

    if not results_summary:
        print("Warning: No results were found to aggregate. Report will be empty.")
        return
        
    # Create a DataFrame and save the report
    report_df = pd.DataFrame(results_summary)
    report_df = report_df.sort_values(by='f1_score', ascending=False).reset_index(drop=True)
    
    report_path = os.path.join(EXPERIMENT_OUTPUT_DIR, 'feature_selection_summary_report.csv')
    report_df.to_csv(report_path, index=False)
    
    print("\n--- Feature Selection Complete ---")
    print("Final Report:")
    print(report_df)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    run_feature_selection() 