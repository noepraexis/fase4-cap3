"""Módulo para visualizações de resultados dos modelos."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(results):
    """
    Plota comparação entre modelos.
    
    Parameters:
        results (dict): Resultados dos modelos
    """
    # Criar DataFrame com métricas
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'Modelo': model_name,
            'Acurácia': result['accuracy'],
            'Precisão': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('Acurácia', ascending=False)
    
    # Plotar comparação
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.2
    
    ax.bar(x - 1.5*width, metrics_df['Acurácia'], width, label='Acurácia')
    ax.bar(x - 0.5*width, metrics_df['Precisão'], width, label='Precisão')
    ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall')
    ax.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score')
    ax.set_title('Comparação de Métricas entre Modelos')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../assets/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_df


def plot_confusion_matrices(results, model_names, class_names=['Kama', 'Rosa', 'Canadian']):
    """
    Plota matrizes de confusão.
    
    Parameters:
        results (dict): Resultados dos modelos
        model_names (list): Lista de modelos para plotar
        class_names (list): Nomes das classes
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        cm = results[model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f'Matriz de Confusão - {model_name}')
        axes[idx].set_xlabel('Predito')
        axes[idx].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig('../assets/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plota importância das características (para Random Forest).
    
    Parameters:
        model: Modelo Random Forest treinado
        feature_names (list): Nomes das características
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
        plt.title('Importância das Características - Random Forest')
        plt.xlabel('Importância')
        plt.tight_layout()
        plt.savefig('../assets/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    else:
        print("O modelo não possui atributo feature_importances_")
        return None


def plot_optimization_results(original_results, optimized_results):
    """
    Plota comparação entre modelos originais e otimizados.
    
    Parameters:
        original_results (dict): Resultados dos modelos originais
        optimized_results (dict): Resultados dos modelos otimizados
    """
    comparison_data = []
    
    for model_name in optimized_results.keys():
        if model_name in original_results:
            comparison_data.append({
                'Modelo': model_name,
                'Acurácia Original': original_results[model_name]['accuracy'],
                'Acurácia Otimizada': optimized_results[model_name]['accuracy'],
                'Melhoria (%)': (optimized_results[model_name]['accuracy'] - 
                               original_results[model_name]['accuracy']) * 100
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plotar comparação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras comparativo
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['Acurácia Original'], width, label='Original')
    ax1.bar(x + width/2, comparison_df['Acurácia Otimizada'], width, label='Otimizada')
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Acurácia')
    ax1.set_title('Comparação: Modelos Originais vs Otimizados')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Modelo'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de melhoria percentual
    ax2.bar(comparison_df['Modelo'], comparison_df['Melhoria (%)'], color='green')
    ax2.set_xlabel('Modelo')
    ax2.set_ylabel('Melhoria (%)')
    ax2.set_title('Melhoria Percentual após Otimização')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../assets/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df


def plot_cross_validation_results(cv_results):
    """
    Plota resultados da validação cruzada.
    
    Parameters:
        cv_results (dict): Resultados da validação cruzada por modelo
    """
    plt.figure(figsize=(10, 6))
    
    models = list(cv_results.keys())
    means = [cv_results[model]['mean'] for model in models]
    stds = [cv_results[model]['std'] for model in models]
    
    x = np.arange(len(models))
    
    plt.bar(x, means, yerr=stds, capsize=10, alpha=0.7)
    plt.xlabel('Modelo')
    plt.ylabel('Acurácia (Validação Cruzada)')
    plt.title('Resultados da Validação Cruzada')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Adicionar valores médios
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../assets/cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()