import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import argparse
from tqdm.auto import tqdm
import textwrap

# List of fingerprint columns to analyze
FINGERPRINT_COLUMNS = [
    "ECFP4", "ECFP6", "FCFP4", "FCFP6", 
    "RDK", "AVALON", "ATOMPAIR", "TOPTOR", "MACCS"
]

def analyze_fingerprint_ranges(filepath, chunk_size=10000):
    """
    Analyzes and prints the min/max value ranges for fingerprint columns in a Parquet file.

    Args:
        filepath (str): Path to the Parquet data file.
        chunk_size (int): Number of rows to read into memory at a time.
    """
    print(f"Analyzing data from: {filepath}")
    
    try:
        pf = pq.ParquetFile(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please provide a valid path to your Parquet dataset.")
        return

    # Initialize dictionaries to store the overall min and max for each column
    min_vals = {col: np.inf for col in FINGERPRINT_COLUMNS}
    max_vals = {col: -np.inf for col in FINGERPRINT_COLUMNS}

    # Get the columns that are actually in the file
    available_columns = [col for col in FINGERPRINT_COLUMNS if col in pf.schema.names]
    if not available_columns:
        print("None of the specified fingerprint columns were found in the file.")
        return

    n_chunks = int(np.ceil(pf.metadata.num_rows / chunk_size))
    pbar = tqdm(total=n_chunks, desc=f'Reading chunks from {filepath}')

    for batch in pf.iter_batches(batch_size=chunk_size, columns=available_columns):
        df = batch.to_pandas()
        for col in available_columns:
            # Drop rows where the fingerprint is null/NaN
            series = df[col].dropna()
            if series.empty:
                continue

            # Process each string of comma-separated values to find min/max.
            # This approach is more memory-efficient than the previous method
            # of splitting and exploding the entire series.
            for val_str in series:
                if not isinstance(val_str, str):
                    continue
                try:
                    # Use a fast list comprehension for conversion
                    nums = [int(x) for x in val_str.split(',')]
                    if not nums:
                        continue
                    
                    # Update overall min/max found so far
                    list_min = min(nums)
                    list_max = max(nums)
                    if list_min < min_vals[col]:
                        min_vals[col] = list_min
                    if list_max > max_vals[col]:
                        max_vals[col] = list_max
                except (ValueError, TypeError):
                    # Skip if a value is not a valid integer string
                    continue
        pbar.update(1)
    pbar.close()

    print("\n--- Fingerprint Value Ranges ---")
    for col in available_columns:
        if np.isinf(min_vals[col]):
            print(f"{col:>10}: No numeric data found.")
        else:
            print(f"{col:>10}: Min={min_vals[col]}, Max={max_vals[col]}")
    print("--------------------------------\n")


def analyze_column_structure(filepath, chunk_size=10000):
    """
    Analyzes and prints the structure and summary statistics for all columns in a Parquet file.

    Args:
        filepath (str): Path to the Parquet data file.
        chunk_size (int): Number of rows to read into memory at a time.
    """
    print(f"\nAnalyzing column structure from: {filepath}")

    try:
        pf = pq.ParquetFile(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please provide a valid path to your Parquet dataset.")
        return

    total_rows = pf.metadata.num_rows
    all_cols = pf.schema.names
    
    # Identify column types
    label_col = 'DELLabel' if 'DELLabel' in all_cols else None
    fp_cols_in_file = [c for c in FINGERPRINT_COLUMNS if c in all_cols]
    other_cols = [c for c in all_cols if c not in fp_cols_in_file and c != label_col]
    
    # Data holders for aggregation across chunks
    data_chunks = {col: [] for col in other_cols}
    if label_col:
        data_chunks[label_col] = []
    fp_len_chunks = {col: [] for col in fp_cols_in_file}
    null_counts = {col: 0 for col in all_cols}

    n_chunks = int(np.ceil(total_rows / chunk_size))
    pbar = tqdm(total=n_chunks, desc=f'Analyzing structure')
    
    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        
        for col in df.columns:
            null_counts[col] += df[col].isnull().sum()
            series = df[col].dropna()
            if series.empty:
                continue

            if col in other_cols or col == label_col:
                data_chunks[col].append(series)
            elif col in fp_cols_in_file:
                # Calculate number of elements in each fingerprint string
                lengths = series.str.split(',').str.len()
                fp_len_chunks[col].append(lengths)
        pbar.update(1)
    pbar.close()

    print("\n--- Column Structure & Statistics ---")
    print(f"Total rows in file: {total_rows}")
    
    for col in sorted(all_cols):
        print(f"\n----- Column: {col} -----")
        non_nulls = total_rows - null_counts[col]
        null_pct = (null_counts[col] / total_rows) * 100 if total_rows > 0 else 0
        print(f"  - Data Type (in Parquet): {pf.schema_arrow.field(col).type}")
        print(f"  - Non-Null Values: {non_nulls} / {total_rows}")
        print(f"  - Null Values: {null_counts[col]} ({null_pct:.2f}%)")
        
        if col in fp_len_chunks and fp_len_chunks[col]:
            all_lengths = pd.concat(fp_len_chunks[col], ignore_index=True)
            print("  - Statistics on Number of Fingerprints per entry:")
            print(textwrap.indent(all_lengths.describe().to_string(), '    '))
        
        elif col in other_cols and data_chunks[col]:
            all_values = pd.concat(data_chunks[col], ignore_index=True)
            if pd.api.types.is_numeric_dtype(all_values):
                print("  - Descriptive Statistics:")
                print(textwrap.indent(all_values.describe().to_string(), '    '))
            else: # For non-numeric object columns
                print("  - Value Counts (Top 5):")
                print(textwrap.indent(all_values.value_counts().nlargest(5).to_string(), '    '))

        elif col == label_col and data_chunks[col]:
            all_labels = pd.concat(data_chunks[col], ignore_index=True)
            print("  - Label Value Counts:")
            print(textwrap.indent(all_labels.value_counts().to_string(), '    '))

    print("\n-------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze value ranges or structure of columns in a Parquet file."
    )
    parser.add_argument(
        "data_path", 
        type=str, 
        help="Path to the Parquet data file."
    )
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['ranges', 'structure'],
        default='ranges',
        help="Type of analysis to perform. 'ranges' for fingerprint values, 'structure' for overall column stats."
    )
    args = parser.parse_args()

    if args.analysis == 'ranges':
        analyze_fingerprint_ranges(args.data_path) 
    elif args.analysis == 'structure':
        analyze_column_structure(args.data_path)
    
    #run with: python crosstalk\data_explore.py "..\data\crosstalk_train (2).parquet" --analysis structure
    