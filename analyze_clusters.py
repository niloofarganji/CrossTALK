import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_cluster_data(input_path, output_dir='Exports'):
    """
    Analyzes the cluster IDs in a Parquet file, prints statistics,
    and generates a histogram of cluster sizes.
    """
    print(f"Loading cluster data from: {input_path}")
    
    # 1. Load the data
    try:
        df = pd.read_parquet(input_path, columns=['cluster_id'])
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return

    if 'cluster_id' not in df.columns:
        print("Error: 'cluster_id' column not found in the file.")
        return

    # 2. Calculate cluster sizes
    cluster_counts = df['cluster_id'].value_counts().sort_values(ascending=False)
    
    # 3. Print summary statistics
    print("\n--- Cluster Size Analysis ---")
    print(f"Total number of molecules: {len(df)}")
    print(f"Total number of unique clusters: {len(cluster_counts)}")
    print("\nStatistics for Cluster Sizes:")
    print(cluster_counts.describe())
    
    print("\nTop 10 Largest Clusters:")
    print(cluster_counts.head(10))

    # Identify single-member clusters (our 'noise' points)
    single_member_clusters = cluster_counts[cluster_counts == 1].count()
    print(f"\nNumber of single-member clusters (outliers): {single_member_clusters}")

    # 4. Generate and save the histogram
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.figure(figsize=(12, 7))
    sns.histplot(data=cluster_counts, log_scale=True, bins=50)
    plt.title('Distribution of Cluster Sizes (Log Scale)')
    plt.xlabel('Cluster Size (Number of Molecules)')
    plt.ylabel('Frequency (Number of Clusters)')
    plt.grid(True, which="both", ls="--")
    
    plot_path = os.path.join(output_dir, 'cluster_size_distribution.png')
    plt.savefig(plot_path)
    
    print(f"\nHistogram of cluster sizes has been saved to: {plot_path}")
    print("--- Analysis Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze the distribution of cluster sizes from a clustered dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default='data/crosstalk_train_with_clusters.parquet',
        help="Path to the input Parquet file containing cluster IDs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='Exports',
        help="Directory to save the output plot."
    )
    args = parser.parse_args()

    analyze_cluster_data(args.input, args.output_dir) 