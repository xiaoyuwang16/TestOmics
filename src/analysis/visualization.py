import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(feature_importances, feature_names, groups):
    plt.figure(figsize=(20, 10))
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    start_idx = 0
    
    for i, (group, count) in enumerate(groups):
        end_idx = start_idx + count
        plt.bar(
            range(start_idx, end_idx),
            feature_importances[start_idx:end_idx],
            color=colors[i],
            label=group
        )
        start_idx = end_idx
    
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.legend()
    plt.show()

def plot_prediction_results(predictions, actuals, confidence_intervals):
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    
    for pred in predictions:
        plt.scatter(
            range(len(pred)), 
            pred, 
            alpha=0.2, 
            c='dodgerblue', 
            edgecolors='none',
            s=50
        )
    
    plt.plot(actuals, label='Actual Values', c='black', linewidth=2)
    plt.plot(
        np.mean(predictions, axis=0),
        label='Predicted Mean',
        c='crimson',
        linewidth=2
    )
    
    plt.fill_between(
        range(len(actuals)),
        confidence_intervals[0],
        confidence_intervals[1],
        color='crimson',
        alpha=0.3
    )
    
    plt.title('Actual vs. Predicted Values with 95% CI')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.show()