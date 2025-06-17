import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class DELDataset(Dataset):
    def __init__(self, parquet_file, fingerprint_cols):
        self.df = pd.read_parquet(parquet_file)
        self.fingerprint_cols = fingerprint_cols
        self.labels = self.df['LABEL'].values

        self.fingerprints = self.df[self.fingerprint_cols]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fingerprints_row = self.fingerprints.iloc[idx]
        
        # Process each fingerprint string and concatenate
        processed_fps = []
        for fp_col in self.fingerprint_cols:
            fp_str = fingerprints_row[fp_col]
            # Convert comma-separated string to numpy array
            fp_arr = np.array(fp_str.split(','), dtype=np.float32)
            processed_fps.append(fp_arr)
        
        # Concatenate all fingerprint arrays into a single feature vector
        feature_vector = np.concatenate(processed_fps)
        
        label = self.labels[idx]
        
        return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    # This is for debugging purposes to find the column names
    # Make sure to adjust the path to your data file.
    file_path = 'data/crosstalk_train (2).parquet'
    df = pd.read_parquet(file_path, engine='pyarrow')
    print(df.columns) 