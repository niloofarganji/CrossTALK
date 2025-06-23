import os
import argparse
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm
import sys

# This block makes the script runnable from anywhere by ensuring the project root
# is on the Python path, making imports absolute and reliable.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Adjust the import path to be relative to the project root
from src.crosstalk.dataset import basic_dataloader
from src.crosstalk.eval import plot_explained_variance

def analyze_svd_components(config, max_components, output_dir):
    """
    Analyzes the explained variance of TruncatedSVD across a range of components.

    Args:
        config (dict): The experiment configuration dictionary.
        max_components (int): The maximum number of SVD components to analyze.
        output_dir (str): The directory to save the analysis plot and results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 1. Load Data (only fingerprints are needed)
    print("\n[1/3] Loading fingerprint data...")
    X_fp, _, _ = basic_dataloader(
        filepath=config['DATA_PATH'],
        fingerprint_cols=config['FINGERPRINT_FEATURES'],
        numeric_cols=None,
        y_col=None,
        max_to_load=config.get('MAX_ROWS')
    )

    # Ensure max_components is not greater than the number of features
    if max_components >= X_fp.shape[1]:
        max_components = X_fp.shape[1] - 1
        print(f"Warning: max_components is too high. Adjusting to {max_components}.")

    # 2. Fit TruncatedSVD
    print(f"\n[2/3] Fitting TruncatedSVD with {max_components} components...")
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    
    # Using tqdm to show progress for the fit, as it can be slow
    # (Note: TruncatedSVD does not have a progress bar built-in, this is a conceptual representation)
    with tqdm(total=1, desc="Fitting SVD") as pbar:
        svd.fit(X_fp)
        pbar.update(1)

    # 3. Analyze and Plot Explained Variance
    print("\n[3/3] Analyzing and plotting explained variance...")
    total_variance = svd.explained_variance_ratio_.sum()
    print(f"Total variance explained by {max_components} components: {total_variance:.4f}")
    
    # Save the plot
    plot_explained_variance(svd, output_dir)
    
    # Also save the raw variance data for detailed analysis
    variance_data = {
        'n_components': list(range(1, max_components + 1)),
        'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
        'cumulative_explained_variance': np.cumsum(svd.explained_variance_ratio_).tolist()
    }
    with open(os.path.join(output_dir, 'svd_variance_analysis.json'), 'w') as f:
        json.dump(variance_data, f, indent=4)
        
    print(f"\nSVD analysis complete. Results saved in '{output_dir}'.")
    print("Inspect the 'svd_explained_variance.png' plot to find the 'elbow' or desired variance capture.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze TruncatedSVD components to find the optimal number for dimensionality reduction."
    )

    # Calculate the absolute path to the default config file
    DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'main.py')

    parser.add_argument(
        "--config", 
        type=str, 
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration file (e.g., 'main.py' or a JSON file)."
    )
    parser.add_argument(
        "--max_components", 
        type=int, 
        default=500,
        help="The maximum number of components to test for SVD."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='Exports/SVD_Analysis',
        help="Directory to save the output plot and results."
    )
    args = parser.parse_args()

    # A simple way to load config from a .py file
    if args.config.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        CONFIG = config_module.CONFIG
    else:
        with open(args.config, 'r') as f:
            CONFIG = json.load(f)

    # Make the data path absolute to ensure it can be found from anywhere
    if not os.path.isabs(CONFIG['DATA_PATH']):
        CONFIG['DATA_PATH'] = os.path.join(PROJECT_ROOT, CONFIG['DATA_PATH'])

    analyze_svd_components(CONFIG, args.max_components, args.output_dir) 