"""Configurações centralizadas do projeto."""

import random
import numpy as np

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

# Paths configuration
ASSETS_DIR = "../assets"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

def set_random_seeds():
    """Configura seeds para reprodutibilidade."""
    random.seed(RANDOM_SEED)
    # scikit-learn usa numpy.random
    np.random.seed(RANDOM_SEED)

def get_asset_path(filename):
    """Retorna caminho completo para arquivo de asset."""
    return f"{ASSETS_DIR}/{filename}"

def print_asset_saved(description, filename):
    """Imprime mensagem padronizada de arquivo salvo."""
    print(f"   ✅ {description}: {get_asset_path(filename)}")