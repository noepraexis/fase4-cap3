#!/usr/bin/env python3
"""Script para analisar detalhadamente as correlações entre características."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_loader import load_seeds_data
from config import FEATURE_NAMES

def analyze_correlations():
    """Analisa detalhadamente as correlações entre características."""
    
    print("="*80)
    print("ANÁLISE DETALHADA DAS CORRELAÇÕES")
    print("="*80)
    
    # Carregar dados
    data = load_seeds_data()
    
    # Calcular matriz de correlação
    correlation_matrix = data[FEATURE_NAMES].corr()
    
    print("\nMatriz de Correlação Completa:")
    print("="*50)
    print(correlation_matrix.round(3))
    
    # Identificar todas as correlações
    print("\n" + "="*50)
    print("ANÁLISE DETALHADA DE CORRELAÇÕES")
    print("="*50)
    
    correlations = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i+1, len(FEATURE_NAMES)):
            corr_value = correlation_matrix.iloc[i, j]
            correlations.append({
                'feature1': FEATURE_NAMES[i],
                'feature2': FEATURE_NAMES[j],
                'correlation': corr_value,
                'abs_correlation': abs(corr_value)
            })
    
    # Ordenar por correlação absoluta
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    # Categorizar correlações
    very_strong = []  # |r| > 0.9
    strong = []       # 0.7 < |r| <= 0.9
    moderate = []     # 0.3 < |r| <= 0.7
    weak = []         # |r| <= 0.3
    
    for corr in correlations:
        abs_corr = corr['abs_correlation']
        if abs_corr > 0.9:
            very_strong.append(corr)
        elif abs_corr > 0.7:
            strong.append(corr)
        elif abs_corr > 0.3:
            moderate.append(corr)
        else:
            weak.append(corr)
    
    # Mostrar correlações por categoria
    print("\n1. CORRELAÇÕES MUITO FORTES (|r| > 0.90):")
    print("-" * 40)
    if very_strong:
        for corr in very_strong:
            direction = "positiva" if corr['correlation'] > 0 else "negativa"
            print(f"• {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f} ({direction})")
    else:
        print("• Nenhuma correlação muito forte encontrada")
    
    print("\n2. CORRELAÇÕES FORTES (0.70 < |r| ≤ 0.90):")
    print("-" * 40)
    if strong:
        for corr in strong:
            direction = "positiva" if corr['correlation'] > 0 else "negativa"
            print(f"• {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f} ({direction})")
    else:
        print("• Nenhuma correlação forte encontrada")
    
    print("\n3. CORRELAÇÕES MODERADAS (0.30 < |r| ≤ 0.70):")
    print("-" * 40)
    if moderate:
        for corr in moderate:
            direction = "positiva" if corr['correlation'] > 0 else "negativa"
            print(f"• {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f} ({direction})")
    else:
        print("• Nenhuma correlação moderada encontrada")
    
    print("\n4. CORRELAÇÕES FRACAS (|r| ≤ 0.30):")
    print("-" * 40)
    if weak:
        for corr in weak:
            direction = "positiva" if corr['correlation'] > 0 else "negativa"
            print(f"• {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f} ({direction})")
    else:
        print("• Nenhuma correlação fraca encontrada")
    
    # Análise de multicolinearidade
    print("\n" + "="*50)
    print("ANÁLISE DE MULTICOLINEARIDADE")
    print("="*50)
    
    print("\nCaracterísticas com correlações problemáticas (|r| > 0.8):")
    problematic = [corr for corr in correlations if corr['abs_correlation'] > 0.8]
    
    if problematic:
        for corr in problematic:
            print(f"⚠️  {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
            print(f"   Risco de multicolinearidade - considerar remoção de uma variável")
    else:
        print("✅ Nenhum problema de multicolinearidade detectado")
    
    # Estatísticas das correlações
    print("\n" + "="*50)
    print("ESTATÍSTICAS DAS CORRELAÇÕES")
    print("="*50)
    
    all_correlations = [corr['correlation'] for corr in correlations]
    abs_correlations = [corr['abs_correlation'] for corr in correlations]
    
    print(f"Total de pares de características: {len(correlations)}")
    print(f"Correlação média: {np.mean(all_correlations):.3f}")
    print(f"Correlação média (valor absoluto): {np.mean(abs_correlations):.3f}")
    print(f"Correlação máxima: {np.max(all_correlations):.3f}")
    print(f"Correlação mínima: {np.min(all_correlations):.3f}")
    print(f"Desvio padrão das correlações: {np.std(all_correlations):.3f}")
    
    # Distribuição das correlações
    print(f"\nDistribuição das correlações:")
    print(f"• Muito fortes (|r| > 0.90): {len(very_strong)} ({len(very_strong)/len(correlations)*100:.1f}%)")
    print(f"• Fortes (0.70 < |r| ≤ 0.90): {len(strong)} ({len(strong)/len(correlations)*100:.1f}%)")
    print(f"• Moderadas (0.30 < |r| ≤ 0.70): {len(moderate)} ({len(moderate)/len(correlations)*100:.1f}%)")
    print(f"• Fracas (|r| ≤ 0.30): {len(weak)} ({len(weak)/len(correlations)*100:.1f}%)")
    
    # Ranking das características mais correlacionadas
    print("\n" + "="*50)
    print("RANKING DE CARACTERÍSTICAS MAIS CORRELACIONADAS")
    print("="*50)
    
    feature_avg_corr = {}
    for feature in FEATURE_NAMES:
        correlations_for_feature = []
        for corr in correlations:
            if corr['feature1'] == feature or corr['feature2'] == feature:
                correlations_for_feature.append(corr['abs_correlation'])
        feature_avg_corr[feature] = np.mean(correlations_for_feature)
    
    # Ordenar por correlação média
    sorted_features = sorted(feature_avg_corr.items(), key=lambda x: x[1], reverse=True)
    
    print("Características ordenadas por correlação média com outras:")
    for i, (feature, avg_corr) in enumerate(sorted_features, 1):
        print(f"{i}. {feature}: {avg_corr:.3f}")
    
    return correlation_matrix, correlations

if __name__ == "__main__":
    analyze_correlations()