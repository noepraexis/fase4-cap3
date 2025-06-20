"""Configurações centralizadas do projeto."""

import random
import numpy as np
from pathlib import Path


def get_project_root():
    """
    Obtém a raiz do projeto de forma robusta.
    Usa a mesma estratégia do utils.py para consistência.
    """
    def is_project_root(path):
        """Verifica se um diretório é a raiz do projeto."""
        path = Path(path)
        indicators = [
            (path / 'src').exists(),
            (path / 'datasets').exists(),
            (path / 'README.md').exists(),
            (path / 'src' / 'config.py').exists(),
            (path / 'src' / 'tests').exists()
        ]
        return sum(indicators) >= 3
    
    # Estratégia 1: A partir do arquivo atual
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if is_project_root(parent):
            return parent
    
    # Estratégia 2: A partir do diretório de trabalho
    current_dir = Path.cwd().resolve()
    for parent in [current_dir] + list(current_dir.parents):
        if is_project_root(parent):
            return parent
    
    # Estratégia 3: Fallback estrutural (config.py está sempre em src/)
    if current_file.parent.name == 'src':
        candidate = current_file.parent.parent
        if is_project_root(candidate):
            return candidate
    
    raise RuntimeError(
        f"Não foi possível encontrar a raiz do projeto.\n"
        f"Arquivo: {current_file}\n"
        f"Diretório atual: {current_dir}"
    )


# Raiz do projeto
PROJECT_ROOT = get_project_root()

# Dataset configuration
FEATURE_NAMES = [
    'area', 'perimeter', 'compactness', 'kernel_length', 
    'kernel_width', 'asymmetry_coefficient', 'kernel_groove_length'
]

VARIETY_NAMES = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}

# Model configuration  
RANDOM_SEED = 42
TEST_SIZE = 0.3
CV_FOLDS = 5

# Paths configuration (caminhos absolutos)
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"

def set_random_seeds():
    """Configura seeds para reprodutibilidade."""
    random.seed(RANDOM_SEED)
    # scikit-learn usa numpy.random
    np.random.seed(RANDOM_SEED)

def get_asset_path(filename):
    """Retorna caminho completo para arquivo de asset."""
    return ASSETS_DIR / filename

def print_asset_saved(description, filename):
    """Imprime mensagem padronizada de arquivo salvo."""
    print(f"   ✅ {description}: {get_asset_path(filename)}")