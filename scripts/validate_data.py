#!/usr/bin/env python3
"""
Script de validação de dados para verificar consistência entre 
src/, tests/, notebooks/ e documentação.
"""

import sys
import os
sys.path.append('src')

from data_loader import load_seeds_data
import pandas as pd
import numpy as np
from config import FEATURE_NAMES

def main():
    """Executa validação completa dos dados do projeto."""
    
    print("="*60)
    print("🔍 VALIDAÇÃO DE INTEGRIDADE DOS DADOS")
    print("="*60)
    
    # Carregar dados
    try:
        df = load_seeds_data()
        print(f"✅ Dataset carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return
    
    # Validações básicas
    print(f"\n📊 INFORMAÇÕES BÁSICAS:")
    print(f"   • Amostras: {len(df)}")
    print(f"   • Características: {len(FEATURE_NAMES)}")
    print(f"   • Classes: {df['variety'].nunique()}")
    print(f"   • Missing values: {df.isnull().sum().sum()}")
    print(f"   • Duplicatas: {df.duplicated().sum()}")
    
    # Distribuição das classes
    print(f"\n🎯 DISTRIBUIÇÃO DAS CLASSES:")
    class_counts = df['variety'].value_counts().sort_index()
    for variety, count in class_counts.items():
        variety_name = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}[variety]
        print(f"   • {variety_name} (Classe {variety}): {count} amostras")
    
    # Coeficientes de Variação
    print(f"\n📈 COEFICIENTES DE VARIAÇÃO:")
    for col in FEATURE_NAMES:
        cv = (df[col].std() / df[col].mean()) * 100
        print(f"   • {col}: {cv:.1f}%")
    
    # Correlações principais
    print(f"\n🔗 CORRELAÇÕES PRINCIPAIS:")
    corr_matrix = df[FEATURE_NAMES].corr()
    correlations = [
        ('area', 'perimeter'),
        ('area', 'kernel_length'),
        ('area', 'kernel_width'),
        ('kernel_length', 'kernel_width'),
        ('perimeter', 'kernel_length')
    ]
    
    for col1, col2 in correlations:
        corr_value = corr_matrix.loc[col1, col2]
        print(f"   • {col1} × {col2}: {corr_value:.3f}")
    
    # Estatísticas por variedade (área)
    print(f"\n🌾 ESTATÍSTICAS POR VARIEDADE (ÁREA):")
    for variety in [1, 2, 3]:
        variety_data = df[df['variety'] == variety]
        mean_area = variety_data['area'].mean()
        std_area = variety_data['area'].std()
        variety_name = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}[variety]
        print(f"   • {variety_name}: μ={mean_area:.3f}, σ={std_area:.3f}")
    
    # Ranges das características
    print(f"\n📏 RANGES DAS CARACTERÍSTICAS:")
    for col in FEATURE_NAMES:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"   • {col}: {min_val:.3f} - {max_val:.3f}")
    
    # Validação de qualidade
    print(f"\n✅ VALIDAÇÃO DE QUALIDADE:")
    
    # Verificar valores extremos
    outliers_count = 0
    for col in FEATURE_NAMES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_count += outliers
        if outliers > 0:
            print(f"   • {col}: {outliers} outliers detectados")
    
    if outliers_count == 0:
        print("   ✅ Nenhum outlier extremo detectado")
    
    # Verificar balanceamento
    class_balance = df['variety'].value_counts()
    is_balanced = all(count == class_balance.iloc[0] for count in class_balance)
    if is_balanced:
        print("   ✅ Classes perfeitamente balanceadas")
    else:
        print("   ⚠️  Classes desbalanceadas")
    
    # Verificar consistência de compacidade
    print(f"\n🧮 VALIDAÇÃO DE COMPACIDADE:")
    calculated_compactness = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
    compactness_diff = np.abs(df['compactness'] - calculated_compactness)
    max_diff = compactness_diff.max()
    mean_diff = compactness_diff.mean()
    
    print(f"   • Diferença máxima: {max_diff:.6f}")
    print(f"   • Diferença média: {mean_diff:.6f}")
    
    if max_diff < 0.001:
        print("   ✅ Compacidade calculada corretamente")
    else:
        print("   ⚠️  Possível inconsistência na compacidade")
    
    print(f"\n" + "="*60)
    print("🎉 VALIDAÇÃO CONCLUÍDA")
    print("="*60)

if __name__ == "__main__":
    main()