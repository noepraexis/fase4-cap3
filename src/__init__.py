"""Sistema de Classificação de Grãos com Machine Learning.

Este pacote implementa um pipeline de aprendizado de máquina para 
classificação automatizada de variedades de grãos de trigo baseado
em suas características físicas.

O sistema utiliza múltiplos algoritmos de classificação (KNN, SVM, 
Random Forest, Naive Bayes, Logistic Regression) com otimização de
hiperparâmetros e validação cruzada para alcançar alta precisão na
identificação das variedades Kama, Rosa e Canadian.

Version:
    1.0.0 - Schierke
    
Modules:
    - config: Configurações centralizadas e constantes
    - data_loader: Carregamento e análise inicial dos dados
    - eda: Análise Exploratória de Dados (EDA)
    - preprocessing: Pré-processamento e normalização
    - models: Treinamento e otimização de modelos
    - visualization: Visualização de resultados
    - utils: Funções utilitárias

Example:
    >>> from src import load_seeds_data, train_all_models
    >>> data = load_seeds_data()
    >>> results = train_all_models(X_train, y_train, X_test, y_test)

Authors:
    - Ana Carolina Belchior (RM565875)
    - Caio Pellegrini (RM566575)
    - Leonardo de Sena (RM563351)
    - Vivian Nascimento Silva Amorim (RM565078)

Institution:
    FIAP - Faculdade de Informática e Administração Paulista
    Project: Da Terra ao Código - Automatizando a Classificação de Grãos

License:
    Creative Commons Atribuição 4.0 Internacional
"""

from typing import Final, List

# Version information
__version__: Final[str] = "1.0.0"
__codename__: Final[str] = "Schierke"
__license__: Final[str] = "Academic"
__status__: Final[str] = "Production"

# Development team
__authors__: Final[List[str]] = [
    "Ana Carolina Belchior (RM565875)",
    "Caio Pellegrini (RM566575)",
    "Leonardo de Sena (RM563351)",
    "Vivian Nascimento Silva Amorim (RM565078)",
]
__institution__: Final[str] = "FIAP - Faculdade de Informática e Administração Paulista"
__project__: Final[str] = "Da Terra ao Código"

# Public API
__all__ = [
    # Version info
    "__version__",
    "__codename__",
    # Data operations
    "load_seeds_data",
    "get_data_info",
    "get_descriptive_statistics",
    # Analysis
    "perform_eda",
    "preprocess_data",
    # Models
    "train_all_models",
    "optimize_model",
    "perform_cross_validation",
    # Configuration
    "set_random_seeds",
    "FEATURE_NAMES",
    "VARIETY_NAMES",
]


def get_version_info() -> dict:
    """Retorna informações completas da versão.
    
    Returns:
        dict: Dicionário contendo version, codename e metadata.
        
    Example:
        >>> info = get_version_info()
        >>> print(f"Running {info['codename']} v{info['version']}")
        Running Schierke v1.0.0
    """
    return {
        "version": __version__,
        "codename": __codename__,
        "authors": __authors__,
        "institution": __institution__,
        "project": __project__,
        "license": __license__,
        "status": __status__,
    }
