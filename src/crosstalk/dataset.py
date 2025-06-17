import pandas as pd
import scipy
import pyarrow.parquet as pq
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import hstack


def basic_dataloader(
    filepath, x_cols, y_col='DELLabel', max_to_load=None, chunk_size=50000
):
    """
    Loads data from a Parquet file, handling multiple feature columns and creating a sparse matrix.

    Args:
        filepath (str): Path to the Parquet file.
        x_cols (list of str): A list of names of the feature columns to be combined.
        y_col (str, optional): Name of the label column. Defaults to 'DELLabel'.
        max_to_load (int, optional): Number of rows to load. If None, loads all rows. Defaults to None.
        chunk_size (int, optional): Number of rows to read at a time from disk. Defaults to 50000.

    Returns:
        X (scipy.sparse.csr_matrix): Combined sparse feature matrix.
        y (np.ndarray or None): Label array if y_col is provided, else None.
    """
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    pf = pq.ParquetFile(filepath)
    columns = x_cols + ([y_col] if y_col is not None else [])
    
    if max_to_load is None:
        max_to_load = pf.metadata.num_rows
    
    list_of_sparse_matrices = [[] for _ in x_cols]
    y_list = []
    loaded = 0

    n_chunks = int(np.ceil(max_to_load / chunk_size))
    pbar = tqdm(total=n_chunks, desc=f'Loading data from {filepath}')
    
    for batch in pf.iter_batches(columns=columns, batch_size=chunk_size):
        batch_df = batch.to_pandas()
        
        current_batch_size = len(batch_df)
        if loaded + current_batch_size > max_to_load:
            remaining = max_to_load - loaded
            batch_df = batch_df.iloc[:remaining]
            current_batch_size = remaining

        if y_col is not None:
            y_list.append(batch_df[y_col].values)

        for i, x_col in enumerate(x_cols):
            # Process each feature column and create a sparse matrix
            exploded = batch_df[x_col].str.split(',', expand=True).astype(np.int8)
            list_of_sparse_matrices[i].append(scipy.sparse.csr_matrix(exploded))
        
        loaded += current_batch_size
        del batch_df
        pbar.update(1)
        if loaded >= max_to_load:
            break
            
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    # Horizontally stack the matrices for each feature to create the final feature matrix
    final_feature_matrices = []
    for mat_list in list_of_sparse_matrices:
        final_feature_matrices.append(scipy.sparse.vstack(mat_list))
    
    X = hstack(final_feature_matrices, format='csr')

    if y_col is not None and y_list:
        y = np.concatenate(y_list)
        return X, y
    else:
        return X
    
    