#!/usr/bin/env python3
"""Script para calcular o Fisher Ratio e outras métricas de separabilidade."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_loader import load_seeds_data
from config import FEATURE_NAMES, VARIETY_NAMES

def calculate_fisher_ratio():
    """Calcula o Fisher Ratio para cada característica."""
    
    print("="*80)
    print("CÁLCULO DO FISHER RATIO E MÉTRICAS DE SEPARABILIDADE")
    print("="*80)
    
    # Carregar dados
    data = load_seeds_data()
    varieties = data['variety'].unique()
    
    print(f"\nDataset: {len(data)} amostras")
    print(f"Variedades: {list(data['variety_name'].unique())}")
    print(f"Features: {len(FEATURE_NAMES)}")
    
    # Calcular Fisher Ratio para cada feature
    print("\n1. CÁLCULO DO FISHER RATIO")
    print("="*50)
    print("\nFórmula: F = σ²(between) / σ²(within)")
    print("onde:")
    print("  σ²(between) = variância entre grupos (inter-classe)")
    print("  σ²(within) = variância dentro dos grupos (intra-classe)")
    
    fisher_ratios = {}
    
    for feature in FEATURE_NAMES:
        print(f"\n{feature.upper()}:")
        print("-" * 40)
        
        # Médias por grupo
        group_means = {}
        group_vars = {}
        group_sizes = {}
        
        for variety in varieties:
            variety_data = data[data['variety'] == variety][feature]
            group_means[variety] = variety_data.mean()
            group_vars[variety] = variety_data.var()
            group_sizes[variety] = len(variety_data)
            
            variety_name = data[data['variety'] == variety]['variety_name'].iloc[0]
            print(f"  {variety_name}:")
            print(f"    n = {group_sizes[variety]}")
            print(f"    μ = {group_means[variety]:.3f}")
            print(f"    σ² = {group_vars[variety]:.3f}")
        
        # Média global
        global_mean = data[feature].mean()
        print(f"\n  Média global: μ_global = {global_mean:.3f}")
        
        # Variância between-group (entre grupos)
        var_between = 0
        for variety in varieties:
            n_i = group_sizes[variety]
            mean_i = group_means[variety]
            var_between += n_i * (mean_i - global_mean) ** 2
        
        var_between = var_between / (len(varieties) - 1)
        print(f"\n  Variância entre grupos (σ²_between):")
        print(f"    Cálculo: Σ[n_i * (μ_i - μ_global)²] / (k-1)")
        print(f"    Valor: {var_between:.3f}")
        
        # Variância within-group (dentro dos grupos)
        var_within = 0
        total_within_samples = 0
        
        for variety in varieties:
            n_i = group_sizes[variety]
            var_i = group_vars[variety]
            var_within += (n_i - 1) * var_i
            total_within_samples += (n_i - 1)
        
        var_within = var_within / total_within_samples
        print(f"\n  Variância dentro dos grupos (σ²_within):")
        print(f"    Cálculo: Σ[(n_i - 1) * σ²_i] / Σ(n_i - 1)")
        print(f"    Valor: {var_within:.3f}")
        
        # Fisher Ratio
        fisher_ratio = var_between / var_within if var_within > 0 else 0
        fisher_ratios[feature] = fisher_ratio
        
        print(f"\n  FISHER RATIO: {fisher_ratio:.2f}")
        
        # Interpretação
        if fisher_ratio > 4:
            interpretation = "EXCELENTE separabilidade"
        elif fisher_ratio > 3:
            interpretation = "MUITO BOA separabilidade"
        elif fisher_ratio > 2:
            interpretation = "BOA separabilidade"
        elif fisher_ratio > 1:
            interpretation = "MODERADA separabilidade"
        else:
            interpretation = "BAIXA separabilidade"
        
        print(f"  Interpretação: {interpretation}")
    
    # Ranking
    print("\n\n2. RANKING DE SEPARABILIDADE")
    print("="*50)
    
    sorted_features = sorted(fisher_ratios.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRanking por Fisher Ratio:")
    print(f"{'Rank':<6} {'Feature':<25} {'Fisher Ratio':<15} {'Classificação'}")
    print("-" * 70)
    
    for i, (feature, ratio) in enumerate(sorted_features, 1):
        if ratio > 4:
            classification = "Excelente"
        elif ratio > 3:
            classification = "Muito boa"
        elif ratio > 2:
            classification = "Boa"
        elif ratio > 1:
            classification = "Moderada"
        else:
            classification = "Baixa"
        
        print(f"{i:<6} {feature:<25} {ratio:<15.2f} {classification}")
    
    # Análise adicional
    print("\n\n3. ANÁLISE DETALHADA DA SEPARABILIDADE")
    print("="*50)
    
    # Distâncias entre centróides
    print("\nDistâncias Euclidianas entre centróides (espaço 7D):")
    
    centroids = {}
    for variety in varieties:
        variety_data = data[data['variety'] == variety][FEATURE_NAMES]
        centroids[variety] = variety_data.mean().values
    
    print("\nCentróides no espaço 7D:")
    for variety in varieties:
        variety_name = data[data['variety'] == variety]['variety_name'].iloc[0]
        print(f"  {variety_name}: {[f'{x:.3f}' for x in centroids[variety]]}")
    
    print("\nMatriz de distâncias:")
    print(f"{'':15} ", end="")
    for v in varieties:
        name = data[data['variety'] == v]['variety_name'].iloc[0]
        print(f"{name:>15}", end="")
    print()
    
    distance_matrix = np.zeros((len(varieties), len(varieties)))
    
    for i, v1 in enumerate(varieties):
        name1 = data[data['variety'] == v1]['variety_name'].iloc[0]
        print(f"{name1:15} ", end="")
        
        for j, v2 in enumerate(varieties):
            if i != j:
                distance = np.sqrt(np.sum((centroids[v1] - centroids[v2])**2))
                distance_matrix[i, j] = distance
                print(f"{distance:15.3f}", end="")
            else:
                print(f"{'---':>15}", end="")
        print()
    
    # Estatísticas das distâncias
    print("\nEstatísticas das distâncias:")
    distances = []
    for i in range(len(varieties)):
        for j in range(i+1, len(varieties)):
            distances.append(distance_matrix[i, j])
    
    print(f"  Distância mínima: {min(distances):.3f}")
    print(f"  Distância máxima: {max(distances):.3f}")
    print(f"  Distância média: {np.mean(distances):.3f}")
    
    # Índice de separabilidade global
    print("\n\n4. ÍNDICE DE SEPARABILIDADE GLOBAL")
    print("="*50)
    
    # Calinski-Harabasz Index
    from sklearn.metrics import calinski_harabasz_score
    
    X = data[FEATURE_NAMES]
    y = data['variety']
    
    ch_index = calinski_harabasz_score(X, y)
    
    print(f"\nÍndice Calinski-Harabasz: {ch_index:.2f}")
    print("\nInterpretação:")
    print("  > 500: Separabilidade excepcional")
    print("  200-500: Separabilidade muito boa")
    print("  100-200: Separabilidade boa")
    print("  50-100: Separabilidade moderada")
    print("  < 50: Separabilidade baixa")
    
    if ch_index > 500:
        ch_interpretation = "EXCEPCIONAL"
    elif ch_index > 200:
        ch_interpretation = "MUITO BOA"
    elif ch_index > 100:
        ch_interpretation = "BOA"
    elif ch_index > 50:
        ch_interpretation = "MODERADA"
    else:
        ch_interpretation = "BAIXA"
    
    print(f"\nClassificação: {ch_interpretation}")
    
    # Cálculo manual para explicação
    print("\nCálculo detalhado do índice CH:")
    
    # Variância intra-cluster
    SSW = 0  # Sum of Squares Within
    for variety in varieties:
        variety_data = data[data['variety'] == variety][FEATURE_NAMES]
        variety_mean = variety_data.mean()
        
        for idx in variety_data.index:
            point = variety_data.loc[idx]
            SSW += np.sum((point - variety_mean) ** 2)
    
    # Variância inter-cluster
    SSB = 0  # Sum of Squares Between
    global_mean = X.mean()
    
    for variety in varieties:
        variety_data = data[data['variety'] == variety][FEATURE_NAMES]
        variety_mean = variety_data.mean()
        n_variety = len(variety_data)
        
        SSB += n_variety * np.sum((variety_mean - global_mean) ** 2)
    
    # Cálculo do índice
    n = len(data)
    k = len(varieties)
    ch_manual = (SSB / (k - 1)) / (SSW / (n - k))
    
    print(f"  SSB (variância entre grupos): {SSB:.2f}")
    print(f"  SSW (variância dentro dos grupos): {SSW:.2f}")
    print(f"  CH = (SSB/(k-1)) / (SSW/(n-k)) = {ch_manual:.2f}")
    
    return fisher_ratios, ch_index

if __name__ == "__main__":
    calculate_fisher_ratio()