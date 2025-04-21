import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from lime import lime_tabular
from src.data.data_loader import load_omix_data
from src.data.data_preprocessing import preprocess_data
from src.models.dnn_model import TestOmix
from src.training.trainer import trainer
from src.analysis.feature_importance import calculate_feature_importance
from src.analysis.visualization import plot_feature_importance, plot_prediction_results
from config.data_config import DATA_CONFIG, MODEL_CONFIG
import matplotlib.pyplot as plt

def ensure_directory_exists(path):
    if isinstance(path, (str, Path)):
        directory = os.path.dirname(str(path)) if os.path.dirname(str(path)) else str(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

def main():
    try:
        print("\nUsing data files:")
        for key, path in DATA_CONFIG['raw_data'].items():
            print(f"{key}: {path}")
        
        for path in [
            DATA_CONFIG['processed_data']['scaled_data'],
            DATA_CONFIG['processed_data']['feature_importance'],
            DATA_CONFIG['output']['model_dir'],
            DATA_CONFIG['output']['results_dir'],
            DATA_CONFIG['output']['visualizations_dir']
        ]:
            ensure_directory_exists(path)
        
        # Loading data
        print("\nLoading data...")
        X, y, feature_names = load_omix_data(
            DATA_CONFIG['raw_data']['rna_path'],
            DATA_CONFIG['raw_data']['protein_path'],
            DATA_CONFIG['raw_data']['metabolomics_path']
        )
        print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
        
        # Data preprocessing
        print("\nPreprocessing data...")
        train_loader, test_loader, X_scaled = preprocess_data(
            X, y, 
            **MODEL_CONFIG['data_params']
        )
        print("Data preprocessing completed.")
        
        # Initialization
        print("\nInitializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TestOmix(
            In_Nodes=X.shape[1],
            **MODEL_CONFIG['model_params']
        ).to(device)
        print("Model initialized.")
        
        # Training
        print("\nStarting model training...")
        best_model_state, train_loss, best_mae = trainer(
            train_loader, 
            test_loader, 
            model,
            **MODEL_CONFIG['training_params']
        )
        print(f"\nTraining completed. Best MAE: {best_mae:.4f}")
        
        model_path = os.path.join(DATA_CONFIG['output']['model_dir'], 'best_model.pth')
        ensure_directory_exists(model_path)
        torch.save(best_model_state, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Feature importance calculation
        print("\nCalculating feature importance...")
        feature_importance = calculate_feature_importance(
            model,
            X_scaled,
            y,
            feature_names,
            device=device
        )
        
        importance_file_path = os.path.join(
            DATA_CONFIG['processed_data']['feature_importance'],
            'feature_importance.csv'  
        )
        ensure_directory_exists(importance_file_path)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        importance_df.to_csv(importance_file_path, index=False)
        print(f"Feature importance saved to {importance_file_path}")

        results_path = os.path.join(
            DATA_CONFIG['output']['results_dir'],
            'training_results.csv'
        )
        ensure_directory_exists(results_path)
        pd.DataFrame({
            'Final_Train_Loss': [train_loss],
            'Best_MAE': [best_mae]
        }).to_csv(results_path, index=False)
        print(f"Training results saved to {results_path}")

        # Visualization
        print("\nGenerating visualizations...")
        viz_dir = DATA_CONFIG['output']['visualizations_dir']
        ensure_directory_exists(viz_dir)

        importance_viz_path = os.path.join(viz_dir, 'feature_importance.png')
        ensure_directory_exists(importance_viz_path)  
        fig = plot_feature_importance(
            feature_importance,
            feature_names,
            MODEL_CONFIG['data_params']
        )
        fig.savefig(importance_viz_path)
        plt.close(fig)

        predictions_viz_path = os.path.join(viz_dir, 'predictions.png')
        ensure_directory_exists(predictions_viz_path)  
        fig = plot_prediction_results(
            model,
            X_scaled,
            y,
            device=device
        )
        fig.savefig(predictions_viz_path)
        plt.close(fig)
        
        print(f"Visualizations saved to {viz_dir}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nProgram completed successfully!")
    

if __name__ == "__main__":
    main()
