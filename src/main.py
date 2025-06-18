"""Script principal para executar a análise completa de classificação de grãos.

Este módulo serve como ponto de entrada para o sistema Schierke de classificação
de grãos, executando o pipeline completo de Machine Learning.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suprime warnings de bibliotecas externas
warnings.filterwarnings('ignore')

# Adiciona o diretório atual ao path para permitir imports relativos
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_seeds_data, get_data_info, get_descriptive_statistics
from eda import perform_eda
from preprocessing import preprocess_data
from models import train_all_models, optimize_model, perform_cross_validation
from visualization import (
    plot_model_comparison, plot_confusion_matrices, plot_feature_importance,
    plot_optimization_results, plot_cross_validation_results
)
from utils import create_directories, save_model, save_results_summary, generate_insights
from config import set_random_seeds

# Import version info
try:
    # Tenta importar do módulo pai se executado como módulo
    from . import __version__, __codename__, __authors__
except ImportError:
    # Se executado como script, importa diretamente
    from __init__ import __version__, __codename__, __authors__


def print_header():
    """Exibe o cabeçalho do sistema com informações de versão."""
    print("="*80)
    print("DA TERRA AO CÓDIGO: CLASSIFICAÇÃO DE GRÃOS COM MACHINE LEARNING")
    print(f"Version {__version__} - Codename: {__codename__}")
    print("-"*80)
    print("Desenvolvido por:")
    for author in __authors__:
        print(f"  • {author}")
    print("="*80)


def main():
    """Executa pipeline completo de análise."""
    
    # Exibe cabeçalho com informações do sistema
    print_header()
    
    # Configurar seeds para reprodutibilidade
    set_random_seeds()
    
    # Criar diretórios necessários
    create_directories()
    
    # 1. Carregar dados
    print("\n1. CARREGANDO DADOS...")
    data = load_seeds_data()  # Usa caminho padrão automático
    data_info = get_data_info(data)
    
    print(f"Dataset carregado: {data_info['shape'][0]} amostras, {data_info['shape'][1]} colunas")
    print(f"Valores ausentes: {data_info['missing_values']}")
    print(f"Distribuição das classes: {data_info['class_distribution']}")
    
    # Estatísticas descritivas
    stats = get_descriptive_statistics(data)
    print("\nEstatísticas Descritivas:")
    print(stats)
    
    # 2. Análise Exploratória de Dados
    print("\n2. REALIZANDO ANÁLISE EXPLORATÓRIA...")
    eda_results = perform_eda(data)
    
    # 3. Pré-processamento
    print("\n3. PRÉ-PROCESSANDO DADOS...")
    processed_data = preprocess_data(data)
    
    # 4. Treinamento de Modelos
    print("\n4. TREINANDO MODELOS...")
    results = train_all_models(
        processed_data['X_train_scaled'],
        processed_data['y_train'],
        processed_data['X_test_scaled'],
        processed_data['y_test']
    )
    
    # Comparação visual
    metrics_df = plot_model_comparison(results)
    print("\nComparação dos Modelos:")
    print(metrics_df)
    
    # Matrizes de confusão dos 3 melhores
    best_models = metrics_df.head(3)['Modelo'].tolist()
    plot_confusion_matrices(results, best_models)
    
    # 5. Otimização de Hiperparâmetros
    print("\n5. OTIMIZANDO HIPERPARÂMETROS...")
    optimized_results = {}
    
    for model_name in best_models[:3]:  # Otimizar os 3 melhores
        if model_name in ['KNN', 'SVM', 'Random Forest']:
            opt_result = optimize_model(
                model_name,
                processed_data['X_train_scaled'],
                processed_data['y_train'],
                processed_data['X_test_scaled'],
                processed_data['y_test']
            )
            if opt_result:
                optimized_results[model_name] = opt_result
                print(f"Melhoria para {model_name}: "
                      f"{(opt_result['accuracy'] - results[model_name]['accuracy'])*100:.2f}%")
    
    # Comparação otimização
    if optimized_results:
        comparison_df = plot_optimization_results(results, optimized_results)
        print("\nComparação: Modelos Originais vs Otimizados")
        print(comparison_df)
    
    # 6. Validação Cruzada
    print("\n6. REALIZANDO VALIDAÇÃO CRUZADA...")
    cv_results = {}
    
    for model_name, opt_result in optimized_results.items():
        cv_result = perform_cross_validation(
            opt_result['model'],
            processed_data['X_train_scaled'],
            processed_data['y_train']
        )
        cv_results[model_name] = cv_result
        print(f"\n{model_name} - CV Score: {cv_result['mean']:.4f} (+/- {cv_result['std']:.4f})")
    
    plot_cross_validation_results(cv_results)
    
    # 7. Análise de Importância das Características
    print("\n7. ANALISANDO IMPORTÂNCIA DAS CARACTERÍSTICAS...")
    feature_importance = None
    if 'Random Forest' in optimized_results:
        feature_importance = plot_feature_importance(
            optimized_results['Random Forest']['model'],
            processed_data['features']
        )
        if feature_importance is not None:
            print("\nImportância das características:")
            print(feature_importance)
    
    # 8. Salvar melhor modelo
    print("\n8. SALVANDO MELHOR MODELO...")
    best_model_name = max(optimized_results.items(), 
                         key=lambda x: x[1]['accuracy'])[0]
    best_model_result = optimized_results[best_model_name]
    
    save_model(
        model=best_model_result['model'],
        scaler=processed_data['scaler'],
        model_name=best_model_name,
        features=processed_data['features'],
        accuracy=best_model_result['accuracy']
    )
    
    # 9. Gerar e salvar insights
    print("\n9. GERANDO INSIGHTS...")
    insights = generate_insights(
        data_info=data_info,
        eda_results=eda_results,
        model_results=optimized_results if optimized_results else results,
        feature_importance=feature_importance
    )
    
    print("\nPRINCIPAIS INSIGHTS:")
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight}")
    
    # Salvar resumo
    save_results_summary(results, optimized_results, insights)
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"Sistema {__codename__} v{__version__} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()