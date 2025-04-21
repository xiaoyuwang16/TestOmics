import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def preprocess_data(X, y, transcriptomics_idx=(0, 597), 
                   proteomics_idx=(597, 1069), 
                   metabolomics_idx=(1069, 1562),
                   test_size=0.2, random_state=42):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values
        
    scaler = StandardScaler()
    
    X_transcriptomics = X[:, transcriptomics_idx[0]:transcriptomics_idx[1]]
    X_proteomics = X[:, proteomics_idx[0]:proteomics_idx[1]]
    X_metabolomics = X[:, metabolomics_idx[0]:metabolomics_idx[1]]
    
    X_transcriptomics_scaled = scaler.fit_transform(X_transcriptomics)
    X_proteomics_scaled = scaler.fit_transform(X_proteomics)
    X_metabolomics_scaled = scaler.fit_transform(X_metabolomics)
    
    X_scaled = np.concatenate([
        X_transcriptomics_scaled,
        X_proteomics_scaled,
        X_metabolomics_scaled
    ], axis=1)
    
    X_noisy1 = X_scaled + np.random.normal(0, 0.01, X_scaled.shape)
    X_noisy2 = X_scaled + np.random.normal(0, 0.01, X_scaled.shape)
    X_noisy3 = X_scaled + np.random.normal(0, 0.01, X_scaled.shape)
    
    X_augmented = np.vstack([X_scaled, X_noisy1, X_noisy2, X_noisy3])
    y_augmented = np.vstack([y, y, y, y])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, 
        y_augmented,
        test_size=test_size,
        random_state=random_state
    )
    
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, test_loader, X_scaled