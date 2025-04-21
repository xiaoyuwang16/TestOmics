import torch
import numpy as np
import pandas as pd
from lime import lime_tabular
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

def calculate_feature_importance(model, X_scaled, y, feature_names, device):
    model.eval()
    
    def predict_fn(x):
        x_tensor = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            return model(x_tensor).cpu().numpy()
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        mode='regression',
        feature_names=feature_names
    )
    
    feature_importances = np.zeros(X_scaled.shape[1])
    num_samples = X_scaled.shape[0]
    
    for i in range(num_samples):
        exp = explainer.explain_instance(
            X_scaled[i], 
            predict_fn,
            num_features=X_scaled.shape[1]
        )
        
        for feature, importance in exp.local_exp[1]:
            feature_importances[feature] += abs(importance)
    
    feature_importances /= num_samples
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    with torch.no_grad():
        base_score = mean_squared_error(
            y_tensor.cpu().numpy(),
            model(X_tensor).cpu().numpy()
        )
    
    permutation_importance = np.zeros(X_scaled.shape[1])
    
    for i in range(X_scaled.shape[1]):
        X_permuted = X_tensor.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(len(X_permuted)), i]
        
        with torch.no_grad():
            permuted_score = mean_squared_error(
                y_tensor.cpu().numpy(),
                model(X_permuted).cpu().numpy()
            )
        
        permutation_importance[i] = base_score - permuted_score
    
    combined_importance = (feature_importances + np.abs(permutation_importance)) / 2
    
    return combined_importance