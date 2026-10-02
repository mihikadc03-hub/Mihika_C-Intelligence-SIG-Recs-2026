import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class CTRDataset(Dataset):
    def __init__(self, dense_X, sparse_X, y):
        self.dense_X = torch.FloatTensor(np.copy(dense_X))
        self.sparse_X = torch.LongTensor(np.copy(sparse_X))
        self.y = torch.FloatTensor(np.copy(y)).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.dense_X[idx], self.sparse_X[idx], self.y[idx]

class CTRDataPreprocessor:
    def __init__(self):
        self.num_cols = [f'integer_feature_{i}' for i in range(1, 14)]
        self.cat_cols = [f'categorical_feature_{i}' for i in range(1, 27)]
        self.scaler = StandardScaler()
        self.cat_mappings = {}
        self.vocab_sizes = []
        
    def fit_transform(self, df):
        # Numeric
        df_num = df[self.num_cols].fillna(0)
        # Apply log transform to handle long tail distribution often seen in CTR datasets
        df_num = np.log1p(np.maximum(df_num, 0)) 
        dense_X = self.scaler.fit_transform(df_num)
        
        # Categorical
        sparse_X = np.zeros((len(df), len(self.cat_cols)), dtype=np.int32)
        for i, col in enumerate(self.cat_cols):
            # Fill NaN with a special string
            col_data = df[col].fillna('<UNK>').astype(str)
            # Create vocabulary (reserve 0 for unknown/unseen during inference)
            unique_vals = col_data.unique()
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            mapping['<UNK>'] = 0
            
            self.cat_mappings[col] = mapping
            self.vocab_sizes.append(len(mapping) + 1) # +1 for safe unk handling if we want
            
            sparse_X[:, i] = col_data.map(mapping).fillna(0).astype(int)
            
        y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        return dense_X, sparse_X, y
        
    def transform(self, df):
        # Numeric
        df_num = df[self.num_cols].fillna(0)
        df_num = np.log1p(np.maximum(df_num, 0))
        dense_X = self.scaler.transform(df_num)
        
        # Categorical
        sparse_X = np.zeros((len(df), len(self.cat_cols)), dtype=np.int32)
        for i, col in enumerate(self.cat_cols):
            col_data = df[col].fillna('<UNK>').astype(str)
            mapping = self.cat_mappings.get(col, {'<UNK>': 0})
            
            # Map known values, map unseen values to 0 (<UNK>)
            sparse_X[:, i] = col_data.map(mapping).fillna(0).astype(int)
            
        y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        return dense_X, sparse_X, y

def load_data(train_path, test_path, val_ratio=0.2, seed=42):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Validation split
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(train_df))
    val_size = int(len(train_df) * val_ratio)
    
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]
    
    val_df = train_df.iloc[val_indices].copy()
    train_df = train_df.iloc[train_indices].copy()
    
    preprocessor = CTRDataPreprocessor()
    
    train_dense, train_sparse, train_y = preprocessor.fit_transform(train_df)
    val_dense, val_sparse, val_y = preprocessor.transform(val_df)
    test_dense, test_sparse, test_y = preprocessor.transform(test_df)
    
    train_dataset = CTRDataset(train_dense, train_sparse, train_y)
    val_dataset = CTRDataset(val_dense, val_sparse, val_y)
    test_dataset = CTRDataset(test_dense, test_sparse, test_y)
    
    return train_dataset, val_dataset, test_dataset, preprocessor
