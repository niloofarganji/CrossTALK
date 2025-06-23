import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

class FingerprintTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that takes a pandas DataFrame of string-formatted
    fingerprints, splits them, and converts them into a single sparse matrix.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Transforms the fingerprint columns into a single sparse matrix.
        
        Args:
            X (pd.DataFrame): DataFrame containing the string-formatted fingerprint columns.

        Returns:
            scipy.sparse.csr_matrix: A single sparse matrix of all combined fingerprints.
        """
        sparse_matrices = []
        for col in X.columns:
            # Split the string and convert to a matrix of integers
            mat = X[col].astype(str).str.split(',', expand=True).to_numpy(dtype=np.int8)
            sparse_matrices.append(csr_matrix(mat))
        
        # Combine all fingerprint columns horizontally
        return hstack(sparse_matrices, format='csr')


def basic_dataloader_pipe(
    filepath, 
    all_feature_cols,
    y_col='DELLabel', 
    max_to_load=None, 
):
    """
    Loads data from a Parquet file for use with a scikit-learn Pipeline.
    This version loads all specified feature columns into a single pandas DataFrame.

    Args:
        filepath (str): Path to the Parquet file.
        all_feature_cols (list of str): All feature columns to load (fingerprint and numeric).
        y_col (str, optional): Name of the label column. Defaults to 'DELLabel'.
        max_to_load (int, optional): Number of rows to load. If None, loads all rows.

    Returns:
        X (pd.DataFrame): DataFrame containing all feature columns.
        y (pd.Series): Series containing the label column.
    """
    
    columns_to_load = all_feature_cols + [y_col]
    
    # For simplicity in this pipeline version, we load the data into pandas directly.
    # The original chunking logic can be re-introduced if memory becomes an issue again.
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath, columns=columns_to_load)

    if max_to_load:
        print(f"Subsetting to {max_to_load} rows.")
        df = df.head(max_to_load)

    # Convert string-based fingerprints to lists of integers for processing if needed,
    # though for this pipeline, we might not need this if the ColumnTransformer can handle it.
    # For now, we assume the pipeline will handle the raw string format or they are pre-processed.
    # This is a simplification point. The original loader did complex parsing.
    # Here we will just pass the raw data through.
    
    X = df[all_feature_cols]
    y = df[y_col]
    
    print("Data loading complete.")
    return X, y 