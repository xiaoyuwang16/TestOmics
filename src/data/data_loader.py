import pandas as pd
import numpy as np

def load_omix_data(rna_path, protein_path, metabolomics_path):

    transcriptomics_data = pd.read_csv(rna_path, index_col=[0,1])
    proteomics_data = pd.read_csv(protein_path, index_col=[0,1])
    metabolomics_data = pd.read_csv(metabolomics_path, index_col=[0])
    
    metabolomics_data_transposed = metabolomics_data.transpose()
    transcriptomics_data = transcriptomics_data.transpose()
    proteomics_data = proteomics_data.transpose().iloc[1:-1]
    
    proteomics_data.columns = proteomics_data.columns.set_levels(
        proteomics_data.columns.levels[0].str.upper(), level=0
    )
    
    current_level_pro = proteomics_data.columns.get_level_values(0)
    proteomics_data.columns = current_level_pro
    current_level_trans = transcriptomics_data.columns.get_level_values(0)
    transcriptomics_data.columns = current_level_trans

    new_index=['Control-1', 'Control-2', 'Control-3', 'S7-1', 'S7-2', 'S7-3']
    proteomics_data.index=new_index
    metabolomics_data_transposed.index=new_index
    
    X = pd.concat([transcriptomics_data, proteomics_data, metabolomics_data_transposed], axis=1)
    feature_names = X.columns.tolist()
    
    y = np.array([[1],[1],[1],[0.5],[0.5],[0.5]])
    
    return X, y, feature_names