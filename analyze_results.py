import pandas as pd
import argparse
import os

def analyze_predictions(filepath):
    """
    Reads a prediction CSV file and counts the number of positive predictions (hits).

    Args:
        filepath (str): The path to the CSV file containing predictions.
    """
    print(f"--- Analyzing Results from: {filepath} ---")

    # 1. Validate input path
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        print("Please provide a valid path to your prediction results file.")
        return

    # 2. Read the CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 3. Validate required column
    if 'predicted_label' not in df.columns:
        print("Error: The CSV file must contain a 'predicted_label' column.")
        return

    # 4. Count the hits and total predictions
    total_predictions = len(df)
    num_hits = df['predicted_label'].sum()  # This works because hits are 1 and non-hits are 0
    hit_rate = (num_hits / total_predictions) * 100 if total_predictions > 0 else 0

    # 5. Print the summary
    print(f"\nTotal molecules analyzed: {total_predictions}")
    print(f"Number of predicted hits (label=1): {num_hits}")
    print(f"Predicted Hit Rate: {hit_rate:.2f}%")
    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyzes a prediction output file to count the number of hits."
    )
    parser.add_argument(
        "results_path", 
        type=str, 
        nargs='?',  # Make the argument optional
        default='Predictions/results1.csv',
        help="Path to the prediction results CSV file. Defaults to 'Predictions/results1.csv'."
    )
    args = parser.parse_args()

    analyze_predictions(args.results_path) 