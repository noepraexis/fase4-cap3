#!/usr/bin/env python3
"""Script para analisar detalhadamente as distribuições das características."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_loader import load_seeds_data
from config import FEATURE_NAMES, VARIETY_NAMES

def analyze_distributions():
    """Analisa detalhadamente as distribuições das características."""
    
    print("="*80)
    print("ANÁLISE DETALHADA DAS DISTRIBUIÇÕES")
    print("="*80)
    
    # Carregar dados
    data = load_seeds_data()
    
    print(f"\nDataset: {data.shape[0]} amostras, {data.shape[1]-1} características")
    print(f"Variedades: {list(data['variety_name'].value_counts().to_dict().keys())}")
    
    # 1. Análise de distribuição por variedade
    print("\n" + "="*50)
    print("1. DISTRIBUIÇÃO DAS VARIEDADES")
    print("="*50)
    variety_counts = data['variety_name'].value_counts()
    for variety, count in variety_counts.items():
        percentage = (count / len(data)) * 100
        print(f"• {variety}: {count} amostras ({percentage:.1f}%)")
    
    # 2. Estatísticas descritivas por característica
    print("\n" + "="*50)
    print("2. ESTATÍSTICAS DESCRITIVAS POR CARACTERÍSTICA")
    print("="*50)
    
    for feature in FEATURE_NAMES:
        print(f"\n--- {feature.upper().replace('_', ' ')} ---")
        
        # Estatísticas gerais
        stats = data[feature].describe()
        print(f"Média: {stats['mean']:.3f}")
        print(f"Mediana: {stats['50%']:.3f}")
        print(f"Desvio Padrão: {stats['std']:.3f}")
        print(f"Mínimo: {stats['min']:.3f}")
        print(f"Máximo: {stats['max']:.3f}")
        print(f"Amplitude: {stats['max'] - stats['min']:.3f}")
        
        # Coeficiente de variação
        cv = (stats['std'] / stats['mean']) * 100
        print(f"Coeficiente de Variação: {cv:.1f}%")
        
        # Análise por variedade
        print("\nPor variedade:")
        for variety in data['variety_name'].unique():
            variety_data = data[data['variety_name'] == variety][feature]
            print(f"  {variety}: μ={variety_data.mean():.3f}, σ={variety_data.std():.3f}, "
                  f"min={variety_data.min():.3f}, max={variety_data.max():.3f}")
    
    # 3. Análise de normalidade
    print("\n" + "="*50)
    print("3. ANÁLISE DE NORMALIDADE (SKEWNESS)")
    print("="*50)
    
    for feature in FEATURE_NAMES:
        skewness = data[feature].skew()
        kurtosis = data[feature].kurtosis()
        
        # Interpretação da assimetria
        if abs(skewness) < 0.5:
            skew_interp = "aproximadamente simétrica"
        elif abs(skewness) < 1:
            skew_interp = "moderadamente assimétrica"
        else:
            skew_interp = "altamente assimétrica"
            
        direction = "à direita" if skewness > 0 else "à esquerda"
        
        print(f"• {feature}: Skewness = {skewness:.3f} ({skew_interp} {direction})")
        print(f"  Kurtosis = {kurtosis:.3f}")
    
    # 4. Identificação de outliers
    print("\n" + "="*50)
    print("4. ANÁLISE DE OUTLIERS (MÉTODO IQR)")
    print("="*50)
    
    total_outliers = 0
    for feature in FEATURE_NAMES:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        print(f"• {feature}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
        if outlier_count > 0:
            print(f"  Limites: [{lower_bound:.3f}, {upper_bound:.3f}]")
            total_outliers += outlier_count
    
    print(f"\nTotal de outliers no dataset: {total_outliers}")
    print(f"Percentual de amostras com outliers: {(total_outliers/len(data)/len(FEATURE_NAMES))*100:.1f}%")
    
    # 5. Separabilidade entre variedades
    print("\n" + "="*50)
    print("5. SEPARABILIDADE ENTRE VARIEDADES")
    print("="*50)
    
    varieties = data['variety_name'].unique()
    for feature in FEATURE_NAMES:
        print(f"\n--- {feature.upper().replace('_', ' ')} ---")
        
        # Calcular sobreposição entre variedades
        variety_ranges = {}
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            variety_ranges[variety] = {
                'min': variety_data.min(),
                'max': variety_data.max(),
                'mean': variety_data.mean(),
                'std': variety_data.std()
            }
        
        # Mostrar separação
        for variety in varieties:
            range_info = variety_ranges[variety]
            print(f"  {variety}: [{range_info['min']:.3f}, {range_info['max']:.3f}] "
                  f"(μ={range_info['mean']:.3f})")
        
        # Calcular distância entre médias
        if len(varieties) == 3:
            v1, v2, v3 = varieties
            dist_12 = abs(variety_ranges[v1]['mean'] - variety_ranges[v2]['mean'])
            dist_13 = abs(variety_ranges[v1]['mean'] - variety_ranges[v3]['mean'])
            dist_23 = abs(variety_ranges[v2]['mean'] - variety_ranges[v3]['mean'])
            
            print(f"  Distâncias entre médias:")
            print(f"    {v1}-{v2}: {dist_12:.3f}")
            print(f"    {v1}-{v3}: {dist_13:.3f}")
            print(f"    {v2}-{v3}: {dist_23:.3f}")

if __name__ == "__main__":
    analyze_distributions()