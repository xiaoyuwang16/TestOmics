import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from matplotlib.patches import Patch

def plot_feature_importance(feature_importance, feature_names, data_params):
    import pandas as pd
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    df['Group'] = 'Other'
    trans_start, trans_end = data_params['transcriptomics_idx']
    prot_start, prot_end = data_params['proteomics_idx']
    meta_start, meta_end = data_params['metabolomics_idx']
    
    df.loc[trans_start:trans_end-1, 'Group'] = 'Transcriptomics'
    df.loc[prot_start:prot_end-1, 'Group'] = 'Proteomics'
    df.loc[meta_start:meta_end-1, 'Group'] = 'Metabolomics'
    
    df = df.sort_values('Importance', ascending=True)
    
    color_map = {
        'Transcriptomics': 'skyblue',
        'Proteomics': 'lightgreen',
        'Metabolomics': 'salmon'
    }
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(df)), df['Importance'])
    
    for i, bar in enumerate(bars):
        bar.set_color(color_map.get(df.iloc[i]['Group'], 'gray'))
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Across Different Omics Data')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=group)
                      for group, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_prediction_results(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, alpha=0.5)
    
    min_val = min(y.min(), predictions.min())
    max_val = max(y.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction Results')
    
    plt.tight_layout()
    
    return plt.gcf()