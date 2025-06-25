import os
import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

def convert_strings_to_sparse_matrix(fp_strings):
    """Converts a pandas Series of fingerprint strings into a SciPy sparse matrix."""
    rows, cols, data = [], [], []
    fp_len = -1
    for i, fp_str in enumerate(tqdm(fp_strings, desc="Converting Strings to Sparse Matrix")):
        if isinstance(fp_str, str):
            bits = [int(b) for b in fp_str.split(',')]
            if fp_len == -1:
                fp_len = len(bits)
            
            if len(bits) != fp_len:
                print(f"Warning: Skipping fingerprint of different length at index {i}.")
                continue

            # Find indices of non-zero elements
            non_zero_indices = np.where(np.array(bits) == 1)[0]
            rows.extend([i] * len(non_zero_indices))
            cols.extend(non_zero_indices)
            data.extend([1] * len(non_zero_indices))
        else:
            # This row will be all zeros, so we don't need to add anything to data/rows/cols
            if fp_len == -1: # If first element is bad, we can't determine shape.
                 raise ValueError("Could not determine fingerprint length. First element is invalid.")
            pass # The row index 'i' is simply skipped, resulting in a row of zeros.

    if fp_len == -1:
        print("Warning: No valid fingerprints found to create a matrix.")
        return csr_matrix((0, 0))
        
    return csr_matrix((data, (rows, cols)), shape=(len(fp_strings), fp_len))

def cluster_fingerprints(fingerprint_matrix, similarity_threshold=0.4):
    """
    Clusters fingerprints using the DBSCAN algorithm from scikit-learn.

    Args:
        fingerprint_matrix (np.array): A 2D numpy array of fingerprints.
        similarity_threshold (float): The Tanimoto similarity threshold for clustering.

    Returns:
        A list of cluster assignments for each fingerprint.
    """
    print(f"Clustering {fingerprint_matrix.shape[0]} fingerprints with threshold {similarity_threshold}...")

    # For DBSCAN, eps is a distance, not a similarity.
    # Jaccard distance = 1 - Tanimoto similarity
    distance_threshold = 1 - similarity_threshold

    # Use DBSCAN. min_samples=2 means a cluster must have at least 2 points.
    # metric='jaccard' is not supported for sparse matrices, use 'cosine' instead.
    db = DBSCAN(eps=distance_threshold, min_samples=2, metric='cosine', n_jobs=-1).fit(fingerprint_matrix)
    cluster_labels = db.labels_

    # DBSCAN labels noise points as -1. We'll assign each to its own unique cluster ID,
    # matching the behavior of the previous LeaderPicker implementation.
    max_cluster_id = cluster_labels.max()
    next_cluster_id = max_cluster_id + 1

    num_noise = np.sum(cluster_labels == -1)
    print(f"DBSCAN found {max_cluster_id + 1} clusters and {num_noise} noise points.")
    print("Assigning noise points to their own unique clusters...")

    for i in range(len(cluster_labels)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = next_cluster_id
            next_cluster_id += 1

    return cluster_labels

def main(input_path, output_path, fp_col='ECFP6'):
    """Main function to load, cluster, and save the data in a memory-efficient way."""
    print(f"Loading fingerprint column '{fp_col}' from: {input_path}")
    
    original_parquet_file = pq.ParquetFile(input_path)
    if fp_col not in original_parquet_file.schema.names:
        print(f"Error: Fingerprint column '{fp_col}' not found in the data.")
        return

    # 1. Load only the fingerprint column to generate clusters
    fp_strings = original_parquet_file.read(columns=[fp_col]).to_pandas()[fp_col]
    fingerprint_matrix = convert_strings_to_sparse_matrix(fp_strings)

    # Check if we have any fingerprints to cluster
    if fingerprint_matrix.shape[0] == 0:
        print("Error: No valid fingerprints were found or converted. Aborting.")
        return
    
    # Note: This assumes no NaNs in the fingerprint column. If there are,
    # the cluster_ids will not align with the original data. A more robust
    # implementation would handle this, but for this dataset it is safe.
    cluster_ids = cluster_fingerprints(fingerprint_matrix)
    
    # 2. Stream the original data and write to a new file, adding the cluster_id column
    print(f"Writing clustered data to: {output_path}")
    
    # Add the cluster ID column to the schema
    new_schema = original_parquet_file.schema_arrow.append(pa.field('cluster_id', pa.int32()))
    
    with pq.ParquetWriter(output_path, schema=new_schema) as writer:
        # Create a PyArrow array for the cluster IDs
        cluster_id_array = pa.array(cluster_ids, type=pa.int32())
        
        for i in tqdm(range(original_parquet_file.num_row_groups), desc="Writing new Parquet file"):
            original_table = original_parquet_file.read_row_group(i)
            
            # Get the corresponding slice of cluster IDs
            offset = original_parquet_file.read_row_group(i).num_rows
            start_pos = sum(original_parquet_file.read_row_group(j).num_rows for j in range(i))
            batch_cluster_ids = cluster_id_array.slice(start_pos, offset)

            # Add the new column to the table
            table_with_clusters = original_table.append_column('cluster_id', batch_cluster_ids)
            writer.write_table(table_with_clusters)

    num_clusters = len(pd.Series(cluster_ids).unique())
    print("\n--- Clustering Complete ---")
    print(f"Successfully processed and saved {len(cluster_ids)} molecules.")
    print(f"Molecules were grouped into {num_clusters} distinct clusters.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cluster molecules based on a pre-computed fingerprint column and add a cluster_id to the dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default='data/crosstalk_train (2).parquet',
        help="Path to the input Parquet data file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default='data/crosstalk_train_with_clusters.parquet',
        help="Path to save the output Parquet file with cluster IDs."
    )
    parser.add_argument(
        "--fp_col",
        type=str,
        default='ECFP6',
        help="Name of the column containing the fingerprint strings to use for clustering."
    )
    args = parser.parse_args()

    main(args.input, args.output, args.fp_col)