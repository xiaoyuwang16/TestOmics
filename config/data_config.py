from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent


DATA_CONFIG = {
    'raw_data': {
        'rna_path': ROOT_DIR / 'data/raw/rna.csv',
        'protein_path': ROOT_DIR / 'data/raw/protein.csv',
        'metabolomics_path': ROOT_DIR / 'data/raw/metabolomics.csv'
    },
    'processed_data': {
        'scaled_data': ROOT_DIR / 'data/processed/scaled_data.csv',
        'feature_importance': ROOT_DIR / 'data/processed/feature_importance.csv'
    }
}