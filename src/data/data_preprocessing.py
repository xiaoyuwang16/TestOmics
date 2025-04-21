import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}

    def __len__(self):
        return len(self.x)

def split_and_scale(X, scaler, indices):
    X_subset = X.iloc[:, indices[0]:indices[1]]
    return scaler.fit_transform(X_subset)

def add_gaussian_noise(X, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def preprocess_data(X, y, transcriptomics_idx, proteomics_idx, metabolomics_idx):
    scaler_transcriptomics = MinMaxScaler()
    scaler_proteomics = MinMaxScaler()
    scaler_metabolomics = MinMaxScaler()
    
    X_transcriptomics_scaled = split_and_scale(X, scaler_transcriptomics, transcriptomics_idx)
    X_proteomics_scaled = split_and_scale(X, scaler_proteomics, proteomics_idx)
    X_metabolomics_scaled = split_and_scale(X, scaler_metabolomics, metabolomics_idx)

    X_scaled = np.concatenate(
        [X_transcriptomics_scaled, X_proteomics_scaled, X_metabolomics_scaled], 
        axis=1
    )
    
    X_train_noisy1 = add_gaussian_noise(X_scaled)
    X_train_noisy2 = add_gaussian_noise(X_scaled)
    X_train_noisy3 = add_gaussian_noise(X_scaled)
    
    X_augmented = np.concatenate([X_scaled, X_train_noisy1, X_train_noisy2, X_train_noisy3])
    y_augmented = np.concatenate([y, y, y, y])
    
    dataset = CustomDataset(X_augmented, y_augmented)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    return dataloader, X_scaled