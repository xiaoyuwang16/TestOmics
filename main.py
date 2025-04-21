from src.data.data_loader import load_omix_data
from src.data.data_preprocessing import preprocess_data
from src.models.dnn_model import TestOmix
from src.training.trainer import trainer
from src.analysis.feature_importance import calculate_feature_importance
from src.analysis.visualization import plot_feature_importance, plot_prediction_results
from config.data_config import DATA_CONFIG

def main():
    config = {
        'model_params': {
            'dropout_rate1': 0.3566765862692761,
            'dropout_rate2': 0.2928632703619797,
            'dim1': 474,
            'dim2': 176,
        },
        'training_params': {
            'lr': 0.0002228301757155853,
            'L2': 0.002039581520209592,
            'num_epochs': 1500,
            'patience': 100,
            'delta': 0.004
        }
    }
    
    # Loading Data
    X, y, feature_names = load_omix_data(
        DATA_CONFIG['raw_data']['rna_path'],
        DATA_CONFIG['raw_data']['protein_path'],
        DATA_CONFIG['raw_data']['metabolomics_path']
    )
    
    # Data Preprocessing
    train_loader, X_scaled = preprocess_data(X, y)
    
    # Model Training
    model = TestOmix(**config['model_params'])
    best_model = trainer(train_loader, test_loader, model, config['training_params'])
    
    # Feature Importance Analysis
    importance_results = calculate_feature_importance(model, X_scaled, y, feature_names)
    
    # Visualization
    plot_feature_importance(importance_results)

if __name__ == "__main__":
    main()