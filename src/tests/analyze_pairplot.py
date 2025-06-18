#!/usr/bin/env python3
"""Script para analisar detalhadamente os dados do pairplot."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data_loader import load_seeds_data
from config import FEATURE_NAMES

def analyze_pairplot():
    """Analisa detalhadamente os dados que compõem o pairplot."""
    
    print("="*80)
    print("ANÁLISE DETALHADA DO PAIRPLOT")
    print("="*80)
    
    # Carregar dados
    data = load_seeds_data()
    varieties = data['variety_name'].unique()
    
    # O pairplot usa as primeiras 4 características
    pairplot_features = FEATURE_NAMES[:4]
    
    print(f"Características no pairplot: {pairplot_features}")
    print(f"Variedades: {varieties}")
    print(f"Total de amostras: {len(data)}")
    
    # Análise das distribuições (diagonal do pairplot)
    print(f"\n{'='*80}")
    print("ANÁLISE DAS DISTRIBUIÇÕES (DIAGONAL)")
    print(f"{'='*80}")
    
    for feature in pairplot_features:
        print(f"\n--- {feature.upper().replace('_', ' ')} ---")
        
        feature_data = data[feature]
        
        # Estatísticas gerais
        print(f"Distribuição geral:")
        print(f"  Média: {feature_data.mean():.3f}")
        print(f"  Mediana: {feature_data.median():.3f}")
        print(f"  Desvio padrão: {feature_data.std():.3f}")
        print(f"  Skewness: {feature_data.skew():.3f}")
        
        # Distribuições por variedade
        print(f"Distribuições por variedade:")
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            print(f"  {variety}:")
            print(f"    Média: {variety_data.mean():.3f}")
            print(f"    Std: {variety_data.std():.3f}")
            print(f"    Min-Max: [{variety_data.min():.3f}, {variety_data.max():.3f}]")
        
        # Análise de separabilidade
        variety_means = []
        variety_stds = []
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            variety_means.append(variety_data.mean())
            variety_stds.append(variety_data.std())
        
        # Distância entre médias
        mean_distances = []
        for i in range(len(varieties)):
            for j in range(i+1, len(varieties)):
                distance = abs(variety_means[i] - variety_means[j])
                mean_distances.append(distance)
                print(f"  Distância {varieties[i]}-{varieties[j]}: {distance:.3f}")
        
        # Coeficiente de separabilidade (distância média entre grupos / variabilidade interna média)
        avg_distance = np.mean(mean_distances)
        avg_std = np.mean(variety_stds)
        separability = avg_distance / avg_std if avg_std > 0 else 0
        print(f"  Coeficiente de separabilidade: {separability:.2f}")
    
    # Análise dos scatter plots (off-diagonal)
    print(f"\n{'='*80}")
    print("ANÁLISE DOS SCATTER PLOTS (RELAÇÕES ENTRE VARIÁVEIS)")
    print(f"{'='*80}")
    
    for i, feature1 in enumerate(pairplot_features):
        for j, feature2 in enumerate(pairplot_features):
            if i < j:  # Apenas half-matrix para evitar duplicação
                print(f"\n--- {feature1.upper()} vs {feature2.upper()} ---")
                
                # Correlação geral
                correlation = data[feature1].corr(data[feature2])
                print(f"Correlação geral: {correlation:.3f}")
                
                # Correlações por variedade
                print("Correlações por variedade:")
                for variety in varieties:
                    variety_data = data[data['variety_name'] == variety]
                    var_correlation = variety_data[feature1].corr(variety_data[feature2])
                    print(f"  {variety}: {var_correlation:.3f}")
                
                # Análise de clusters visuais
                print("Análise de agrupamento:")
                
                for variety in varieties:
                    variety_data = data[data['variety_name'] == variety]
                    x_data = variety_data[feature1]
                    y_data = variety_data[feature2]
                    
                    # Centro do cluster
                    center_x = x_data.mean()
                    center_y = y_data.mean()
                    
                    # Dispersão do cluster
                    dispersion_x = x_data.std()
                    dispersion_y = y_data.std()
                    
                    # Compacidade do cluster (raio médio)
                    distances_from_center = np.sqrt((x_data - center_x)**2 + (y_data - center_y)**2)
                    cluster_radius = distances_from_center.mean()
                    
                    print(f"  {variety}:")
                    print(f"    Centro: ({center_x:.3f}, {center_y:.3f})")
                    print(f"    Dispersão: ({dispersion_x:.3f}, {dispersion_y:.3f})")
                    print(f"    Raio médio: {cluster_radius:.3f}")
                
                # Distâncias entre centros dos clusters
                centers = {}
                for variety in varieties:
                    variety_data = data[data['variety_name'] == variety]
                    centers[variety] = (
                        variety_data[feature1].mean(),
                        variety_data[feature2].mean()
                    )
                
                print("Distâncias entre centros dos clusters:")
                for i, var1 in enumerate(varieties):
                    for var2 in varieties[i+1:]:
                        center1 = centers[var1]
                        center2 = centers[var2]
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        print(f"  {var1} ↔ {var2}: {distance:.3f}")
                
                # Sobreposição entre clusters
                print("Análise de sobreposição:")
                
                # Calcular bounding boxes para cada variedade
                bboxes = {}
                for variety in varieties:
                    variety_data = data[data['variety_name'] == variety]
                    bboxes[variety] = {
                        'x_min': variety_data[feature1].min(),
                        'x_max': variety_data[feature1].max(),
                        'y_min': variety_data[feature2].min(),
                        'y_max': variety_data[feature2].max()
                    }
                
                for i, var1 in enumerate(varieties):
                    for var2 in varieties[i+1:]:
                        bbox1 = bboxes[var1]
                        bbox2 = bboxes[var2]
                        
                        # Verificar sobreposição
                        x_overlap = max(0, min(bbox1['x_max'], bbox2['x_max']) - max(bbox1['x_min'], bbox2['x_min']))
                        y_overlap = max(0, min(bbox1['y_max'], bbox2['y_max']) - max(bbox1['y_min'], bbox2['y_min']))
                        
                        if x_overlap > 0 and y_overlap > 0:
                            overlap_area = x_overlap * y_overlap
                            
                            # Área dos bounding boxes
                            area1 = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
                            area2 = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])
                            
                            # Porcentagem de sobreposição
                            overlap_pct1 = (overlap_area / area1) * 100
                            overlap_pct2 = (overlap_area / area2) * 100
                            
                            print(f"  {var1} ↔ {var2}: {overlap_pct1:.1f}% e {overlap_pct2:.1f}% de sobreposição")
                        else:
                            print(f"  {var1} ↔ {var2}: Sem sobreposição")
    
    # Resumo de separabilidade
    print(f"\n{'='*80}")
    print("RESUMO: SEPARABILIDADE NO ESPAÇO MULTIDIMENSIONAL")
    print(f"{'='*80}")
    
    # Calcular separabilidade multidimensional
    print("\nAnálise de separabilidade global:")
    
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        
        # Centro do cluster multidimensional
        center = variety_data.mean()
        
        # Distâncias de cada ponto ao centro
        distances = []
        for idx in variety_data.index:
            point = variety_data.loc[idx]
            distance = np.sqrt(sum((point[feat] - center[feat])**2 for feat in pairplot_features))
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        max_distance = max(distances)
        
        print(f"{variety}:")
        print(f"  Centro: {[f'{center[feat]:.3f}' for feat in pairplot_features]}")
        print(f"  Raio médio: {avg_distance:.3f}")
        print(f"  Raio máximo: {max_distance:.3f}")
    
    # Distâncias entre centros multidimensionais
    print("\nDistâncias entre centros multidimensionais:")
    centers_multi = {}
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        centers_multi[variety] = variety_data.mean()
    
    for i, var1 in enumerate(varieties):
        for var2 in varieties[i+1:]:
            center1 = centers_multi[var1]
            center2 = centers_multi[var2]
            distance = np.sqrt(sum((center1[feat] - center2[feat])**2 for feat in pairplot_features))
            print(f"  {var1} ↔ {var2}: {distance:.3f}")
    
    # Qualidade de separação
    print("\nQualidade de separação:")
    
    # Calcular inércia intra-cluster e inter-cluster
    intra_cluster_variance = 0
    total_samples = 0
    
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        center = variety_data.mean()
        
        for idx in variety_data.index:
            point = variety_data.loc[idx]
            distance_sq = sum((point[feat] - center[feat])**2 for feat in pairplot_features)
            intra_cluster_variance += distance_sq
            total_samples += 1
    
    intra_cluster_variance /= total_samples
    
    # Calcular variância inter-cluster
    global_center = data[pairplot_features].mean()
    inter_cluster_variance = 0
    
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        variety_center = variety_data.mean()
        variety_size = len(variety_data)
        
        distance_sq = sum((variety_center[feat] - global_center[feat])**2 for feat in pairplot_features)
        inter_cluster_variance += variety_size * distance_sq
    
    inter_cluster_variance /= total_samples
    
    # Calinski-Harabasz Index (higher is better)
    calinski_harabasz = (inter_cluster_variance / intra_cluster_variance) * ((total_samples - len(varieties)) / (len(varieties) - 1))
    
    print(f"  Variância intra-cluster: {intra_cluster_variance:.3f}")
    print(f"  Variância inter-cluster: {inter_cluster_variance:.3f}")
    print(f"  Índice Calinski-Harabasz: {calinski_harabasz:.2f}")
    
    if calinski_harabasz > 100:
        separation_quality = "EXCELENTE"
    elif calinski_harabasz > 50:
        separation_quality = "BOA"
    elif calinski_harabasz > 20:
        separation_quality = "MODERADA"
    else:
        separation_quality = "RUIM"
    
    print(f"  Qualidade de separação: {separation_quality}")

if __name__ == "__main__":
    analyze_pairplot()