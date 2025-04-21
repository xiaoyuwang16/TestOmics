import torch
import numpy as np
from lime import lime_tabular
from sklearn.metrics import mean_squared_error

def calculate_feature_importance(model, X_scaled, y, feature_names, device):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        mode='regression',
        feature_names=feature_names
    )
    
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(x_tensor)
        return preds.cpu().numpy()
    
    feature_importances = np.zeros((X_scaled.shape[0], X_scaled.shape[1]))
    for i in range(X_scaled.shape[0]):
        exp = explainer.explain_instance(
            X_scaled[i], 
            predict_fn, 
            num_features=X_scaled.shape[1]
        )
        exp_map = {exp.domain_mapper.feature_names[j]: exp.local_exp[1][j][1] 
                  for j in range(X_scaled.shape[1])}
        feature_importances[i] = [exp_map[feature] for feature in feature_names]
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        original_preds = model(X_tensor)
        original_score = mean_squared_error(
            y_tensor.cpu().numpy(), 
            original_preds.cpu().numpy()
        )
    
    permutation_importance = []
    for i in range(X_scaled.shape[1]):
        X_permuted = X_tensor.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.shape[0]), i]
        permuted_preds = model(X_permuted)
        permuted_score = mean_squared_error(
            y_tensor.cpu().numpy(), 
            permuted_preds.cpu().numpy()
        )
        permutation_importance.append(original_score - permuted_score)
    
    return feature_importances, permutation_importance