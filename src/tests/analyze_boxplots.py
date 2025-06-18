#!/usr/bin/env python3
"""Script para analisar detalhadamente os boxplots por variedade."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_loader import load_seeds_data
from config import FEATURE_NAMES

def analyze_boxplots():
    """Analisa detalhadamente os dados dos boxplots por variedade."""
    
    print("="*80)
    print("ANÁLISE DETALHADA DOS BOXPLOTS POR VARIEDADE")
    print("="*80)
    
    # Carregar dados
    data = load_seeds_data()
    varieties = data['variety_name'].unique()
    
    print(f"Variedades analisadas: {varieties}")
    print(f"Total de características: {len(FEATURE_NAMES)}")
    
    # Análise detalhada por característica
    for feature in FEATURE_NAMES:
        print(f"\n{'='*80}")
        print(f"CARACTERÍSTICA: {feature.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        # Estatísticas por variedade
        variety_stats = {}
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            
            # Calcular estatísticas dos boxplots
            Q1 = variety_data.quantile(0.25)
            Q2 = variety_data.quantile(0.50)  # Mediana
            Q3 = variety_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Limites dos whiskers (1.5 * IQR)
            lower_whisker = Q1 - 1.5 * IQR
            upper_whisker = Q3 + 1.5 * IQR
            
            # Ajustar whiskers para valores reais
            actual_lower = variety_data[variety_data >= lower_whisker].min()
            actual_upper = variety_data[variety_data <= upper_whisker].max()
            
            # Identificar outliers
            outliers = variety_data[(variety_data < lower_whisker) | (variety_data > upper_whisker)]
            
            variety_stats[variety] = {
                'min': variety_data.min(),
                'Q1': Q1,
                'median': Q2,
                'Q3': Q3,
                'max': variety_data.max(),
                'IQR': IQR,
                'lower_whisker_calc': lower_whisker,
                'upper_whisker_calc': upper_whisker,
                'lower_whisker_actual': actual_lower,
                'upper_whisker_actual': actual_upper,
                'outliers': outliers.tolist(),
                'outlier_count': len(outliers),
                'mean': variety_data.mean(),
                'std': variety_data.std()
            }
        
        # Mostrar estatísticas detalhadas
        print("\nEstatísticas dos Boxplots:")
        print("-" * 60)
        
        for variety in varieties:
            stats = variety_stats[variety]
            print(f"\n{variety}:")
            print(f"  Mínimo: {stats['min']:.3f}")
            print(f"  Q1 (25%): {stats['Q1']:.3f}")
            print(f"  Mediana (Q2): {stats['median']:.3f}")
            print(f"  Q3 (75%): {stats['Q3']:.3f}")
            print(f"  Máximo: {stats['max']:.3f}")
            print(f"  IQR: {stats['IQR']:.3f}")
            print(f"  Whisker inferior: {stats['lower_whisker_actual']:.3f}")
            print(f"  Whisker superior: {stats['upper_whisker_actual']:.3f}")
            print(f"  Outliers: {stats['outlier_count']} valores")
            if stats['outliers']:
                print(f"    Valores: {[round(x, 3) for x in stats['outliers']]}")
            print(f"  Média: {stats['mean']:.3f}")
            print(f"  Desvio padrão: {stats['std']:.3f}")
        
        # Análise comparativa entre variedades
        print(f"\nAnálise Comparativa para {feature}:")
        print("-" * 60)
        
        # Ordenar variedades por mediana
        sorted_varieties = sorted(varieties, key=lambda x: variety_stats[x]['median'])
        
        print("Ordenação por mediana (menor → maior):")
        for i, variety in enumerate(sorted_varieties, 1):
            median = variety_stats[variety]['median']
            print(f"  {i}. {variety}: {median:.3f}")
        
        # Calcular sobreposição de IQRs
        print("\nSobreposição de Intervalos Interquartílicos (IQR):")
        for i, var1 in enumerate(varieties):
            for var2 in varieties[i+1:]:
                stats1 = variety_stats[var1]
                stats2 = variety_stats[var2]
                
                # Verificar sobreposição
                overlap_start = max(stats1['Q1'], stats2['Q1'])
                overlap_end = min(stats1['Q3'], stats2['Q3'])
                
                if overlap_start <= overlap_end:
                    overlap_size = overlap_end - overlap_start
                    total_range = max(stats1['Q3'], stats2['Q3']) - min(stats1['Q1'], stats2['Q1'])
                    overlap_percentage = (overlap_size / total_range) * 100
                    print(f"  {var1} ↔ {var2}: {overlap_percentage:.1f}% de sobreposição")
                else:
                    print(f"  {var1} ↔ {var2}: Sem sobreposição de IQR")
        
        # Identificar a característica mais discriminativa
        print("\nCapacidade Discriminativa:")
        
        # Calcular separação entre medianas
        medians = [variety_stats[var]['median'] for var in varieties]
        median_range = max(medians) - min(medians)
        
        # Calcular variabilidade média interna
        avg_iqr = np.mean([variety_stats[var]['IQR'] for var in varieties])
        
        # Ratio separação/variabilidade (maior = mais discriminativo)
        discrimination_ratio = median_range / avg_iqr if avg_iqr > 0 else 0
        
        print(f"  Amplitude das medianas: {median_range:.3f}")
        print(f"  IQR médio: {avg_iqr:.3f}")
        print(f"  Ratio discriminativo: {discrimination_ratio:.2f}")
        
        if discrimination_ratio > 2:
            discriminative_level = "ALTA"
        elif discrimination_ratio > 1:
            discriminative_level = "MODERADA"
        else:
            discriminative_level = "BAIXA"
        
        print(f"  Capacidade discriminativa: {discriminative_level}")
        
        # Identificar variedade com maior/menor dispersão
        iqrs = {var: variety_stats[var]['IQR'] for var in varieties}
        most_dispersed = max(iqrs, key=iqrs.get)
        least_dispersed = min(iqrs, key=iqrs.get)
        
        print(f"  Maior dispersão: {most_dispersed} (IQR = {iqrs[most_dispersed]:.3f})")
        print(f"  Menor dispersão: {least_dispersed} (IQR = {iqrs[least_dispersed]:.3f})")
    
    # Resumo geral de discriminação
    print(f"\n{'='*80}")
    print("RESUMO: RANKING DE CARACTERÍSTICAS MAIS DISCRIMINATIVAS")
    print(f"{'='*80}")
    
    discrimination_scores = {}
    
    for feature in FEATURE_NAMES:
        variety_medians = []
        variety_iqrs = []
        
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            variety_medians.append(variety_data.median())
            variety_iqrs.append(variety_data.quantile(0.75) - variety_data.quantile(0.25))
        
        median_range = max(variety_medians) - min(variety_medians)
        avg_iqr = np.mean(variety_iqrs)
        
        discrimination_scores[feature] = median_range / avg_iqr if avg_iqr > 0 else 0
    
    # Ordenar por capacidade discriminativa
    sorted_features = sorted(discrimination_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCaracterísticas ordenadas por capacidade discriminativa:")
    for i, (feature, score) in enumerate(sorted_features, 1):
        print(f"{i}. {feature}: {score:.2f}")
    
    return variety_stats

if __name__ == "__main__":
    analyze_boxplots()