"""Módulo para pré-processamento dos dados."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED, TEST_SIZE


def prepare_features_target(data):
    """
    Separa características e rótulos.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        
    Returns:
        tuple: (X, y) - características e rótulos
    """
    from config import FEATURE_NAMES
    features = FEATURE_NAMES
    
    X = data[features]
    y = data['variety']
    
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, verbose=True):
    """
    Divide dados em treino e teste.
    
    Parameters:
        X (pd.DataFrame): Características
        y (pd.Series): Rótulos
        test_size (float): Proporção do conjunto de teste
        random_state (int): Seed para reprodutibilidade
        verbose (bool): Se deve imprimir informações detalhadas
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"✅ Dados divididos: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, verbose=True):
    """
    Normaliza as características usando StandardScaler.
    
    Parameters:
        X_train (pd.DataFrame): Características de treino
        X_test (pd.DataFrame): Características de teste
        verbose (bool): Se deve imprimir informações detalhadas
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print("✅ Características normalizadas (μ≈0, σ≈1)")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(data, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Pipeline completo de pré-processamento.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        test_size (float): Proporção do conjunto de teste
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        dict: Dicionário com todos os dados processados
    """
    # Preparar características e rótulos
    X, y = prepare_features_target(data)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Normalizar características
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'features': list(X.columns)
    }