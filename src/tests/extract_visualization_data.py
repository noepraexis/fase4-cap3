#!/usr/bin/env python3
"""
Script para extrair dados exatos das visualizações conforme o notebook.
Reproduz os mesmos valores para documentação precisa.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from data_loader import load_seeds_data
from config import FEATURE_NAMES, VARIETY_NAMES, RANDOM_SEED

# Constantes para análise
ASSETS_DIR = '../assets'
TEST_DATA_DIR = '../assets/test_data'
Q1_PERCENTILE = 0.25
Q2_PERCENTILE = 0.50  # Mediana
Q3_PERCENTILE = 0.75
IQR_MULTIPLIER = 1.5
WHISKER_MULTIPLIER = 1.5

# Thresholds de correlação
VERY_STRONG_THRESHOLD = 0.90
STRONG_THRESHOLD = 0.70
MODERATE_THRESHOLD = 0.30
MULTICOLLINEARITY_THRESHOLD = 0.8

# Thresholds de qualidade
EXCELLENT_CALINSKI_THRESHOLD = 100
GOOD_CALINSKI_THRESHOLD = 50

# Nomes de arquivos
DISTRIBUTIONS_FILE = 'distributions_data.json'
CORRELATION_FILE = 'correlation_data.json'
BOXPLOT_FILE = 'boxplot_data.json'
PAIRPLOT_FILE = 'pairplot_data.json'

# Número de características no pairplot
PAIRPLOT_FEATURES_COUNT = 4

# Separadores e formatação
SEPARATOR_80 = "=" * 80
SEPARATOR_60 = "=" * 60

def extract_distributions_data():
    """Extrai dados exatos das distribuições conforme notebook."""
    
    data = load_seeds_data()
    
    print(SEPARATOR_80)
    print("EXTRAÇÃO DE DADOS DAS DISTRIBUIÇÕES")
    print(SEPARATOR_80)
    
    # Dados básicos do dataset
    dataset_info = {
        'total_samples': len(data),
        'features_count': len(FEATURE_NAMES),
        'varieties': list(data['variety_name'].unique()),
        'samples_per_variety': data['variety_name'].value_counts().to_dict()
    }
    
    print(f"Dataset: {dataset_info['total_samples']} amostras, {dataset_info['features_count']} características")
    print(f"Variedades: {dataset_info['varieties']}")
    print(f"Distribuição: {dataset_info['samples_per_variety']}")
    
    # Estatísticas por característica (8 gráficos)
    distributions_data = {}
    
    for feature in FEATURE_NAMES:
        feature_data = data[feature]
        
        # Estatísticas gerais
        stats = {
            'mean': float(feature_data.mean()),
            'median': float(feature_data.median()),
            'std': float(feature_data.std()),
            'min': float(feature_data.min()),
            'max': float(feature_data.max()),
            'skewness': float(feature_data.skew()),
            'kurtosis': float(feature_data.kurtosis()),
            'coefficient_variation': float((feature_data.std() / feature_data.mean()) * 100)
        }
        
        # Estatísticas por variedade
        variety_stats = {}
        for variety in data['variety_name'].unique():
            variety_data = data[data['variety_name'] == variety][feature]
            variety_stats[variety] = {
                'mean': float(variety_data.mean()),
                'std': float(variety_data.std()),
                'min': float(variety_data.min()),
                'max': float(variety_data.max()),
                'count': int(len(variety_data))
            }
        
        # Análise de outliers (método IQR)
        Q1 = feature_data.quantile(Q1_PERCENTILE)
        Q3 = feature_data.quantile(Q3_PERCENTILE)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outlier_info = {
            'count': len(outliers),
            'percentage': float((len(outliers) / len(data)) * 100),
            'values': outliers[feature].tolist() if len(outliers) > 0 else [],
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        
        distributions_data[feature] = {
            'general_stats': stats,
            'variety_stats': variety_stats,
            'outliers': outlier_info
        }
        
        print(f"\n{feature.upper().replace('_', ' ')}:")
        print(f"  CV: {stats['coefficient_variation']:.1f}%")
        print(f"  Outliers: {outlier_info['count']} ({outlier_info['percentage']:.1f}%)")
    
    # Gráfico das variedades (8º gráfico)
    variety_distribution = {
        'counts': data['variety_name'].value_counts().to_dict(),
        'percentages': (data['variety_name'].value_counts(normalize=True) * 100).round(1).to_dict()
    }
    
    distributions_data['variety_distribution'] = variety_distribution
    
    # Salvar dados extraídos
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    distributions_file_path = os.path.join(TEST_DATA_DIR, DISTRIBUTIONS_FILE)
    with open(distributions_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_info': dataset_info,
            'distributions': distributions_data,
            'extraction_date': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Dados das distribuições salvos em: {distributions_file_path}")
    return distributions_data

def extract_correlation_data():
    """Extrai dados exatos da matriz de correlação conforme notebook."""
    
    data = load_seeds_data()
    
    print(f"\n{SEPARATOR_80}")
    print("EXTRAÇÃO DE DADOS DA MATRIZ DE CORRELAÇÃO")
    print(SEPARATOR_80)
    
    # Calcular matriz de correlação
    correlation_matrix = data[FEATURE_NAMES].corr()
    
    # Extrair todas as correlações
    correlations = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i+1, len(FEATURE_NAMES)):
            corr_value = correlation_matrix.iloc[i, j]
            correlations.append({
                'feature1': FEATURE_NAMES[i],
                'feature2': FEATURE_NAMES[j],
                'correlation': float(corr_value),
                'abs_correlation': float(abs(corr_value))
            })
    
    # Categorizar correlações
    very_strong = [c for c in correlations if c['abs_correlation'] > VERY_STRONG_THRESHOLD]
    strong = [c for c in correlations if STRONG_THRESHOLD < c['abs_correlation'] <= VERY_STRONG_THRESHOLD]
    moderate = [c for c in correlations if MODERATE_THRESHOLD < c['abs_correlation'] <= STRONG_THRESHOLD]
    weak = [c for c in correlations if c['abs_correlation'] <= MODERATE_THRESHOLD]
    
    # Estatísticas das correlações
    all_corr_values = [c['correlation'] for c in correlations]
    abs_corr_values = [c['abs_correlation'] for c in correlations]
    
    correlation_stats = {
        'total_pairs': len(correlations),
        'mean_correlation': float(np.mean(all_corr_values)),
        'mean_abs_correlation': float(np.mean(abs_corr_values)),
        'max_correlation': float(np.max(all_corr_values)),
        'min_correlation': float(np.min(all_corr_values)),
        'std_correlation': float(np.std(all_corr_values))
    }
    
    # Problemas de multicolinearidade
    problematic = [c for c in correlations if c['abs_correlation'] > MULTICOLLINEARITY_THRESHOLD]
    
    correlation_data = {
        'matrix': correlation_matrix.round(3).to_dict(),
        'categorized': {
            'very_strong': very_strong,
            'strong': strong,
            'moderate': moderate,
            'weak': weak
        },
        'statistics': correlation_stats,
        'multicollinearity': {
            'problematic_pairs': problematic,
            'count': len(problematic)
        },
        'distribution': {
            'very_strong_pct': float(len(very_strong) / len(correlations) * 100),
            'strong_pct': float(len(strong) / len(correlations) * 100),
            'moderate_pct': float(len(moderate) / len(correlations) * 100),
            'weak_pct': float(len(weak) / len(correlations) * 100)
        }
    }
    
    print(f"Correlações muito fortes (>{VERY_STRONG_THRESHOLD}): {len(very_strong)} ({len(very_strong)/len(correlations)*100:.1f}%)")
    print(f"Correlações fortes ({STRONG_THRESHOLD}-{VERY_STRONG_THRESHOLD}): {len(strong)} ({len(strong)/len(correlations)*100:.1f}%)")
    print(f"Correlações moderadas ({MODERATE_THRESHOLD}-{STRONG_THRESHOLD}): {len(moderate)} ({len(moderate)/len(correlations)*100:.1f}%)")
    print(f"Correlações fracas (≤{MODERATE_THRESHOLD}): {len(weak)} ({len(weak)/len(correlations)*100:.1f}%)")
    print(f"Problemas multicolinearidade: {len(problematic)}")
    
    # Salvar dados extraídos
    correlation_file_path = os.path.join(TEST_DATA_DIR, CORRELATION_FILE)
    with open(correlation_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'correlation_analysis': correlation_data,
            'extraction_date': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Dados da correlação salvos em: {correlation_file_path}")
    return correlation_data

def extract_boxplot_data():
    """Extrai dados exatos dos boxplots conforme notebook."""
    
    data = load_seeds_data()
    varieties = data['variety_name'].unique()
    
    print(f"\n{SEPARATOR_80}")
    print("EXTRAÇÃO DE DADOS DOS BOXPLOTS")
    print(SEPARATOR_80)
    
    boxplot_data = {}
    discrimination_scores = {}
    
    for feature in FEATURE_NAMES:
        feature_analysis = {}
        
        # Estatísticas por variedade
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            
            Q1 = variety_data.quantile(Q1_PERCENTILE)
            Q2 = variety_data.quantile(Q2_PERCENTILE)  # Mediana
            Q3 = variety_data.quantile(Q3_PERCENTILE)
            IQR = Q3 - Q1
            
            # Whiskers
            lower_whisker = Q1 - WHISKER_MULTIPLIER * IQR
            upper_whisker = Q3 + WHISKER_MULTIPLIER * IQR
            actual_lower = variety_data[variety_data >= lower_whisker].min()
            actual_upper = variety_data[variety_data <= upper_whisker].max()
            
            # Outliers
            outliers = variety_data[(variety_data < lower_whisker) | (variety_data > upper_whisker)]
            
            feature_analysis[variety] = {
                'min': float(variety_data.min()),
                'Q1': float(Q1),
                'median': float(Q2),
                'Q3': float(Q3),
                'max': float(variety_data.max()),
                'IQR': float(IQR),
                'lower_whisker': float(actual_lower),
                'upper_whisker': float(actual_upper),
                'outliers': outliers.tolist(),
                'outlier_count': len(outliers),
                'mean': float(variety_data.mean()),
                'std': float(variety_data.std())
            }
        
        # Calcular capacidade discriminativa
        variety_medians = [feature_analysis[var]['median'] for var in varieties]
        variety_iqrs = [feature_analysis[var]['IQR'] for var in varieties]
        
        median_range = max(variety_medians) - min(variety_medians)
        avg_iqr = np.mean(variety_iqrs)
        
        discrimination_ratio = median_range / avg_iqr if avg_iqr > 0 else 0
        discrimination_scores[feature] = float(discrimination_ratio)
        
        # Análise de sobreposição
        overlap_analysis = {}
        for i, var1 in enumerate(varieties):
            for var2 in varieties[i+1:]:
                Q1_1, Q3_1 = feature_analysis[var1]['Q1'], feature_analysis[var1]['Q3']
                Q1_2, Q3_2 = feature_analysis[var2]['Q1'], feature_analysis[var2]['Q3']
                
                overlap_start = max(Q1_1, Q1_2)
                overlap_end = min(Q3_1, Q3_2)
                
                if overlap_start <= overlap_end:
                    overlap_size = overlap_end - overlap_start
                    total_range = max(Q3_1, Q3_2) - min(Q1_1, Q1_2)
                    overlap_percentage = (overlap_size / total_range) * 100
                    overlap_analysis[f"{var1}_{var2}"] = float(overlap_percentage)
                else:
                    overlap_analysis[f"{var1}_{var2}"] = 0.0
        
        boxplot_data[feature] = {
            'variety_statistics': feature_analysis,
            'discrimination_ratio': discrimination_ratio,
            'overlap_analysis': overlap_analysis,
            'median_range': float(median_range),
            'avg_iqr': float(avg_iqr)
        }
        
        print(f"{feature}: discrimination_ratio = {discrimination_ratio:.2f}")
    
    # Ranking de discriminação
    sorted_discrimination = sorted(discrimination_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_data = {
        'boxplot_analysis': boxplot_data,
        'discrimination_ranking': sorted_discrimination,
        'varieties': varieties.tolist()
    }
    
    # Salvar dados extraídos
    boxplot_file_path = os.path.join(TEST_DATA_DIR, BOXPLOT_FILE)
    with open(boxplot_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'boxplot_data': final_data,
            'extraction_date': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nRanking discriminativo:")
    for i, (feature, score) in enumerate(sorted_discrimination, 1):
        print(f"  {i}. {feature}: {score:.2f}")
    
    print(f"\n✅ Dados dos boxplots salvos em: {boxplot_file_path}")
    return final_data

def extract_pairplot_data():
    """Extrai dados exatos do pairplot conforme notebook."""
    
    data = load_seeds_data()
    varieties = data['variety_name'].unique()
    
    # Pairplot usa as 4 primeiras características
    pairplot_features = FEATURE_NAMES[:PAIRPLOT_FEATURES_COUNT]
    
    print(f"\n{SEPARATOR_80}")
    print("EXTRAÇÃO DE DADOS DO PAIRPLOT")
    print(SEPARATOR_80)
    
    print(f"Características no pairplot: {pairplot_features}")
    
    # Análise das distribuições (diagonal)
    diagonal_data = {}
    for feature in pairplot_features:
        feature_data = data[feature]
        
        # Estatísticas gerais
        general_stats = {
            'mean': float(feature_data.mean()),
            'median': float(feature_data.median()),
            'std': float(feature_data.std()),
            'skewness': float(feature_data.skew())
        }
        
        # Por variedade
        variety_stats = {}
        variety_means = []
        variety_stds = []
        
        for variety in varieties:
            variety_data = data[data['variety_name'] == variety][feature]
            stats = {
                'mean': float(variety_data.mean()),
                'std': float(variety_data.std()),
                'min': float(variety_data.min()),
                'max': float(variety_data.max())
            }
            variety_stats[variety] = stats
            variety_means.append(stats['mean'])
            variety_stds.append(stats['std'])
        
        # Coeficiente de separabilidade
        mean_distances = []
        for i in range(len(varieties)):
            for j in range(i+1, len(varieties)):
                distance = abs(variety_means[i] - variety_means[j])
                mean_distances.append(distance)
        
        avg_distance = np.mean(mean_distances)
        avg_std = np.mean(variety_stds)
        separability = avg_distance / avg_std if avg_std > 0 else 0
        
        diagonal_data[feature] = {
            'general_stats': general_stats,
            'variety_stats': variety_stats,
            'separability_coefficient': float(separability),
            'mean_distances': mean_distances,
            'avg_distance': float(avg_distance),
            'avg_std': float(avg_std)
        }
    
    # Análise dos scatter plots (off-diagonal)
    scatter_data = {}
    for i, feature1 in enumerate(pairplot_features):
        for j, feature2 in enumerate(pairplot_features):
            if i < j:  # Apenas half-matrix
                pair_key = f"{feature1}_vs_{feature2}"
                
                # Correlação geral
                correlation = data[feature1].corr(data[feature2])
                
                # Correlações por variedade
                variety_correlations = {}
                for variety in varieties:
                    variety_data = data[data['variety_name'] == variety]
                    var_correlation = variety_data[feature1].corr(variety_data[feature2])
                    variety_correlations[variety] = float(var_correlation)
                
                # Análise de clusters
                cluster_analysis = {}
                for variety in varieties:
                    variety_data = data[data['variety_name'] == variety]
                    x_data = variety_data[feature1]
                    y_data = variety_data[feature2]
                    
                    center_x = x_data.mean()
                    center_y = y_data.mean()
                    dispersion_x = x_data.std()
                    dispersion_y = y_data.std()
                    
                    distances_from_center = np.sqrt((x_data - center_x)**2 + (y_data - center_y)**2)
                    cluster_radius = distances_from_center.mean()
                    
                    cluster_analysis[variety] = {
                        'center': [float(center_x), float(center_y)],
                        'dispersion': [float(dispersion_x), float(dispersion_y)],
                        'radius': float(cluster_radius)
                    }
                
                # Distâncias entre centros
                center_distances = {}
                centers = {variety: cluster_analysis[variety]['center'] for variety in varieties}
                for i_var, var1 in enumerate(varieties):
                    for var2 in varieties[i_var+1:]:
                        center1 = centers[var1]
                        center2 = centers[var2]
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                        center_distances[f"{var1}_{var2}"] = float(distance)
                
                scatter_data[pair_key] = {
                    'general_correlation': float(correlation),
                    'variety_correlations': variety_correlations,
                    'cluster_analysis': cluster_analysis,
                    'center_distances': center_distances
                }
    
    # Análise multidimensional
    multidim_data = {}
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        center = variety_data.mean()
        
        distances = []
        for idx in variety_data.index:
            point = variety_data.loc[idx]
            distance = np.sqrt(sum((point[feat] - center[feat])**2 for feat in pairplot_features))
            distances.append(distance)
        
        multidim_data[variety] = {
            'center': center.tolist(),
            'avg_radius': float(np.mean(distances)),
            'max_radius': float(max(distances))
        }
    
    # Distâncias entre centros multidimensionais
    multi_center_distances = {}
    centers_multi = {variety: multidim_data[variety]['center'] for variety in varieties}
    for i, var1 in enumerate(varieties):
        for var2 in varieties[i+1:]:
            center1 = centers_multi[var1]
            center2 = centers_multi[var2]
            distance = np.sqrt(sum((center1[i] - center2[i])**2 for i in range(len(pairplot_features))))
            multi_center_distances[f"{var1}_{var2}"] = float(distance)
    
    # Índice Calinski-Harabasz
    total_samples = 0
    intra_cluster_variance = 0
    
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        center = variety_data.mean()
        
        for idx in variety_data.index:
            point = variety_data.loc[idx]
            distance_sq = sum((point[feat] - center[feat])**2 for feat in pairplot_features)
            intra_cluster_variance += distance_sq
            total_samples += 1
    
    intra_cluster_variance /= total_samples
    
    # Variância inter-cluster
    global_center = data[pairplot_features].mean()
    inter_cluster_variance = 0
    
    for variety in varieties:
        variety_data = data[data['variety_name'] == variety][pairplot_features]
        variety_center = variety_data.mean()
        variety_size = len(variety_data)
        
        distance_sq = sum((variety_center[feat] - global_center[feat])**2 for feat in pairplot_features)
        inter_cluster_variance += variety_size * distance_sq
    
    inter_cluster_variance /= total_samples
    
    # Calinski-Harabasz Index
    calinski_harabasz = (inter_cluster_variance / intra_cluster_variance) * ((total_samples - len(varieties)) / (len(varieties) - 1))
    
    quality_metrics = {
        'intra_cluster_variance': float(intra_cluster_variance),
        'inter_cluster_variance': float(inter_cluster_variance),
        'calinski_harabasz_index': float(calinski_harabasz),
        'total_samples': total_samples
    }
    
    pairplot_data = {
        'features': pairplot_features,
        'diagonal_analysis': diagonal_data,
        'scatter_analysis': scatter_data,
        'multidimensional_analysis': multidim_data,
        'multi_center_distances': multi_center_distances,
        'quality_metrics': quality_metrics
    }
    
    # Salvar dados extraídos
    pairplot_file_path = os.path.join(TEST_DATA_DIR, PAIRPLOT_FILE)
    with open(pairplot_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pairplot_data': pairplot_data,
            'extraction_date': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Índice Calinski-Harabasz: {calinski_harabasz:.2f}")
    
    if calinski_harabasz > EXCELLENT_CALINSKI_THRESHOLD:
        quality_level = "EXCELENTE"
    elif calinski_harabasz > GOOD_CALINSKI_THRESHOLD:
        quality_level = "BOA"
    else:
        quality_level = "MODERADA"
    
    print(f"Qualidade de separação: {quality_level}")
    
    print(f"\n✅ Dados do pairplot salvos em: {pairplot_file_path}")
    return pairplot_data

def main():
    """Função principal para extrair todos os dados das visualizações."""
    
    print("🔍 INICIANDO EXTRAÇÃO COMPLETA DOS DADOS DAS VISUALIZAÇÕES")
    print(SEPARATOR_80)
    
    try:
        # Criar diretório test_data se não existir
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
        # Extrair dados de cada tipo de visualização
        distributions = extract_distributions_data()
        correlations = extract_correlation_data()
        boxplots = extract_boxplot_data()
        pairplot = extract_pairplot_data()
        
        # Resumo final
        print(f"\n{SEPARATOR_80}")
        print("RESUMO DA EXTRAÇÃO")
        print(SEPARATOR_80)
        
        # Gerar caminhos dos arquivos
        files_generated = [
            os.path.join(TEST_DATA_DIR, DISTRIBUTIONS_FILE),
            os.path.join(TEST_DATA_DIR, CORRELATION_FILE),
            os.path.join(TEST_DATA_DIR, BOXPLOT_FILE),
            os.path.join(TEST_DATA_DIR, PAIRPLOT_FILE)
        ]
        
        print("✅ Arquivos gerados:")
        for file_path in files_generated:
            print(f"  • {file_path}")
        
        # Constantes para resumo
        TOTAL_DISTRIBUTION_PLOTS = len(FEATURE_NAMES) + 1  # 7 características + 1 gráfico de variedades
        
        print("\n🎯 Dados extraídos para documentação:")
        print(f"  • {TOTAL_DISTRIBUTION_PLOTS} gráficos de distribuições analisados")
        print(f"  • {len(correlations['categorized']['very_strong'])} correlações muito fortes identificadas")
        print(f"  • {len(FEATURE_NAMES)} características analisadas em boxplots")
        print(f"  • {len(pairplot['features'])} características no pairplot")
        print(f"  • Índice Calinski-Harabasz: {pairplot['quality_metrics']['calinski_harabasz_index']:.2f}")
        
        print("\n✅ EXTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("Os dados estão prontos para atualização da documentação.")
        
    except Exception as e:
        print(f"❌ Erro durante extração: {e}")
        raise

if __name__ == "__main__":
    main()