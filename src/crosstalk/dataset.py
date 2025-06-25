import pandas as pd
import scipy
import pyarrow.parquet as pq
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import hstack, csr_matrix


def parse_fingerprints_to_sparse(fp_series):
    """
    Efficiently converts a pandas Series of comma-separated fingerprint strings
    into a SciPy CSR sparse matrix.
    """
    rows, cols, data = [], [], []
    fp_len = -1

    for i, fp_str in enumerate(fp_series):
        if isinstance(fp_str, str):
            bits = [int(b) for b in fp_str.split(',')]
            if fp_len == -1:
                fp_len = len(bits)
            
            if len(bits) != fp_len:
                # In a real scenario, might need more robust error handling
                continue 

            non_zero_indices = np.where(np.array(bits) == 1)[0]
            rows.extend([i] * len(non_zero_indices))
            cols.extend(non_zero_indices)
            data.extend([1] * len(non_zero_indices))

    # If fp_len was never set, it means no valid fingerprints were found
    if fp_len == -1:
        if len(fp_series) > 0:
             # Return an empty matrix with the correct number of rows but 0 columns
             return csr_matrix((len(fp_series), 0))
        else: # No data at all
             return csr_matrix((0,0))
    
    return csr_matrix((data, (rows, cols)), shape=(len(fp_series), fp_len))


def basic_dataloader(
    filepath, 
    fingerprint_cols, 
    numeric_cols=None,
    y_col='DELLabel', 
    max_to_load=None, 
    chunk_size=50000
):
    """
    Loads data from a Parquet file, handling fingerprint, numeric, and label columns.

    Args:
        filepath (str): Path to the Parquet file.
        fingerprint_cols (list of str): A list of fingerprint column names to be combined.
        numeric_cols (list of str, optional): A list of numeric column names. Defaults to None.
        y_col (str, optional): Name of the label column. Defaults to 'DELLabel'.
        max_to_load (int, optional): Number of rows to load. If None, loads all rows.
        chunk_size (int, optional): Number of rows to read at a time from disk.

    Returns:
        X_fp (scipy.sparse.csr_matrix): Combined sparse fingerprint matrix.
        X_num (np.ndarray or None): Combined numeric feature matrix.
        y (np.ndarray or None): Label array if y_col is provided, else None.
    """
    if isinstance(fingerprint_cols, str):
        fingerprint_cols = [fingerprint_cols]
    
    numeric_cols = numeric_cols or []
    
    # Determine all columns to load from the file
    columns_to_load = fingerprint_cols + numeric_cols
    if y_col:
        columns_to_load.append(y_col)

    pf = pq.ParquetFile(filepath)
    if max_to_load is None:
        max_to_load = pf.metadata.num_rows
    
    # Initialize lists to hold the data chunks
    fp_chunks = [[] for _ in fingerprint_cols]
    num_chunks = []
    y_list = []
    loaded = 0

    n_chunks = int(np.ceil(max_to_load / chunk_size))
    pbar = tqdm(total=n_chunks, desc=f'Loading data from {filepath}')
    
    for batch in pf.iter_batches(columns=columns_to_load, batch_size=chunk_size):
        batch_df = batch.to_pandas()
        
        current_batch_size = len(batch_df)
        if loaded + current_batch_size > max_to_load:
            remaining = max_to_load - loaded
            batch_df = batch_df.iloc[:remaining]
            current_batch_size = remaining

        # Process and append fingerprint data
        for i, fp_col in enumerate(fingerprint_cols):
            # OLD: exploded = batch_df[fp_col].str.split(',', expand=True).astype(np.int8)
            # NEW: Efficiently convert to sparse without intermediate dense matrix
            fp_sparse_chunk = parse_fingerprints_to_sparse(batch_df[fp_col])
            fp_chunks[i].append(fp_sparse_chunk)
        
        # Process and append numeric data
        if numeric_cols:
            num_chunks.append(batch_df[numeric_cols].values)

        if y_col:
            y_list.append(batch_df[y_col].values)
        
        loaded += current_batch_size
        del batch_df
        pbar.update(1)
        if loaded >= max_to_load:
            break
            
    pbar.close()

    # --- Assemble final matrices ---
    # Vertically stack the lists of chunk matrices, then horizontally stack the final fp matrices
    final_fp_matrices = [scipy.sparse.vstack(chunks) for chunks in fp_chunks]
    X_fp = hstack(final_fp_matrices, format='csr') if final_fp_matrices else None

    # Concatenate numeric data chunks
    X_num = np.vstack(num_chunks) if num_chunks else None
    
    # Concatenate label data chunks
    y = np.concatenate(y_list) if y_list else None

    return X_fp, X_num, y
    
    