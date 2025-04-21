from pathlib import Path
import os

# 设置根目录和数据目录
ROOT_DIR = "/your_path"  
DATA_DIRS = {
    'rna_file': "data/rna.csv",     
    'protein_file': "data/proteins.csv",              
    'metabolomics_file': "data/metabolomics.csv"         
}

# 自动生成完整路径配置
DATA_CONFIG = {
    'raw_data': {
        'rna_path': str(Path(ROOT_DIR) / DATA_DIRS['rna_file']),
        'protein_path': str(Path(ROOT_DIR) / DATA_DIRS['protein_file']),
        'metabolomics_path': str(Path(ROOT_DIR) / DATA_DIRS['metabolomics_file'])
    },
    'processed_data': {
        'scaled_data': str(Path(ROOT_DIR) / 'data/processed/scaled_data.csv'),
        'feature_importance': str(Path(ROOT_DIR) / 'data/processed')  # 改为目录路径
    },
    'output': {
        'model_dir': str(Path(ROOT_DIR) / 'models'),
        'results_dir': str(Path(ROOT_DIR) / 'results'),
        'visualizations_dir': str(Path(ROOT_DIR) / 'results/visualizations')
    }
}

# 模型配置保持不变
MODEL_CONFIG = {
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
    },
    'data_params': {
        'transcriptomics_idx': (0, 597),
        'proteomics_idx': (597, 1069),
        'metabolomics_idx': (1069, 1562)
    }
}