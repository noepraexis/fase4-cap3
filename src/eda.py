"""Módulo para Análise Exploratória de Dados (EDA)."""

import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_NAMES, get_asset_path, print_asset_saved


def setup_visualization():
    """Configura parâmetros de visualização."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_distributions(data, features):
    """
    Plota histogramas das características.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        features (list): Lista de características para plotar
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        axes[idx].hist(data[feature], bins=20, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribuição de {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequência')
    
    # Gráfico de barras para as variedades
    axes[-2].bar(data['variety_name'].value_counts().index, 
                 data['variety_name'].value_counts().values)
    axes[-2].set_title('Distribuição das Variedades')
    axes[-2].set_xlabel('Variedade')
    axes[-2].set_ylabel('Quantidade')
    
    # Remover o último subplot vazio
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(get_asset_path('distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print_asset_saved("Distribuições", 'distributions.png')


def plot_boxplots_by_variety(data, features):
    """
    Plota boxplots das características por variedade.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        features (list): Lista de características para plotar
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        data.boxplot(column=feature, by='variety_name', ax=axes[idx])
        axes[idx].set_title(f'{feature} por Variedade')
        axes[idx].set_xlabel('Variedade')
        axes[idx].set_ylabel(feature)
    
    # Remover subplots vazios
    for i in range(len(features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Boxplots das Características por Variedade', y=1.02)
    plt.tight_layout()
    plt.savefig(get_asset_path('boxplots_by_variety.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print_asset_saved("Boxplots", 'boxplots_by_variety.png')


def plot_correlation_matrix(data, features):
    """
    Plota matriz de correlação das características.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        features (list): Lista de características
    """
    correlation_matrix = data[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlação das Características', fontsize=16)
    plt.tight_layout()
    plt.savefig(get_asset_path('correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print_asset_saved("Correlação", 'correlation_matrix.png')
    
    return correlation_matrix


def plot_pairplot(data, features):
    """
    Plota pairplot para visualizar relações entre variáveis.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        features (list): Lista de características (recomenda-se usar no máximo 4)
    """
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(data, hue='variety_name', vars=features[:4], 
                             diag_kind='kde', palette='husl')
    pairplot.fig.suptitle('Relações entre as Principais Características', y=1.02)
    plt.savefig(get_asset_path('pairplot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print_asset_saved("Pairplot", 'pairplot.png')


def perform_eda(data):
    """
    Executa análise exploratória completa.
    
    Parameters:
        data (pd.DataFrame): DataFrame com os dados
        
    Returns:
        dict: Dicionário com resultados da análise
    """
    setup_visualization()
    
    print("📊 Gerando visualizações EDA...")
    
    # Plotar visualizações
    plot_distributions(data, FEATURE_NAMES)
    plot_boxplots_by_variety(data, FEATURE_NAMES)
    correlation_matrix = plot_correlation_matrix(data, FEATURE_NAMES)
    plot_pairplot(data, FEATURE_NAMES)
    
    # Análise de correlações fortes
    strong_correlations = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i+1, len(FEATURE_NAMES)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                strong_correlations.append({
                    'feature1': FEATURE_NAMES[i],
                    'feature2': FEATURE_NAMES[j],
                    'correlation': corr_value
                })
    
    if strong_correlations:
        print(f"\n🔍 Correlações fortes identificadas:")
        for corr in strong_correlations:
            print(f"   • {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'strong_correlations': strong_correlations
    }