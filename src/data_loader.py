"""Módulo para carregamento e análise inicial dos dados."""

import pandas as pd
from pathlib import Path


def load_seeds_data(filepath=None):
    """
    Carrega o dataset de sementes.
    
    Parameters:
        filepath (str, optional): Caminho para o arquivo de dados.
                                Se None, usa caminho padrão baseado na localização do arquivo.
        
    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    if filepath is None:
        # Caminho absoluto baseado na localização deste arquivo
        project_root = Path(__file__).parent.parent
        filepath = project_root / "datasets" / "seeds_dataset.txt"
    
    from config import FEATURE_NAMES, VARIETY_NAMES
    column_names = FEATURE_NAMES + ['variety']
    
    data = pd.read_csv(filepath, sep=r'\s+', header=None, names=column_names)
    
    # Adicionar nomes das variedades
    data['variety_name'] = data['variety'].map(VARIETY_NAMES)
    
    return data


def get_data_info(data):
    """
    Retorna informações básicas sobre o dataset.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        
    Returns:
        dict: Dicionário com informações do dataset
    """
    info = {
        'shape': data.shape,
        'columns': list(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'class_distribution': data['variety'].value_counts().to_dict(),
        'class_proportions': data['variety'].value_counts(normalize=True).to_dict()
    }
    
    return info


def get_descriptive_statistics(data):
    """
    Calcula estatísticas descritivas para as características.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        
    Returns:
        pd.DataFrame: Estatísticas descritivas
    """
    from config import FEATURE_NAMES
    features = FEATURE_NAMES
    
    stats = data[features].describe()
    stats = stats.round(4)
    
    return stats