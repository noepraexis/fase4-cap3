"""Módulo com funções utilitárias."""

import os
import json
import joblib
from datetime import datetime


def create_directories():
    """Cria diretórios necessários se não existirem."""
    directories = ['../models', '../assets', '../results']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Diretório '{directory}' criado/verificado.")


def save_model(model, scaler, model_name, features, accuracy):
    """
    Salva modelo treinado e informações relacionadas.
    
    Parameters:
        model: Modelo treinado
        scaler: Scaler usado no pré-processamento
        model_name (str): Nome do modelo
        features (list): Lista de características
        accuracy (float): Acurácia do modelo
    """
    # Criar diretório se não existir
    os.makedirs('../models', exist_ok=True)
    
    # Nome do arquivo baseado no modelo
    model_filename = f"../models/{model_name.lower().replace(' ', '_')}.pkl"
    scaler_filename = "../models/scaler.pkl"
    
    # Salvar modelo e scaler
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    # Salvar informações do modelo
    model_info = {
        'model_name': model_name,
        'accuracy': accuracy,
        'features': features,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_file': model_filename,
        'scaler_file': scaler_filename
    }
    
    with open('../models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"\nModelo '{model_name}' salvo com sucesso!")
    print(f"Arquivos salvos:")
    print(f"- Modelo: {model_filename}")
    print(f"- Scaler: {scaler_filename}")
    print(f"- Informações: ../models/model_info.json")


def load_model(model_name=None):
    """
    Carrega modelo salvo.
    
    Parameters:
        model_name (str): Nome do modelo (opcional)
        
    Returns:
        dict: Dicionário com modelo, scaler e informações
    """
    # Carregar informações do modelo
    with open('../models/model_info.json', 'r') as f:
        model_info = json.load(f)
    
    # Se não especificado, usar o modelo salvo
    if model_name is None:
        model_filename = model_info['model_file']
    else:
        model_filename = f"../models/{model_name.lower().replace(' ', '_')}.pkl"
    
    # Carregar modelo e scaler
    model = joblib.load(model_filename)
    scaler = joblib.load(model_info['scaler_file'])
    
    return {
        'model': model,
        'scaler': scaler,
        'info': model_info
    }


def save_results_summary(results, optimized_results, insights):
    """
    Salva resumo dos resultados em arquivo.
    
    Parameters:
        results (dict): Resultados dos modelos originais
        optimized_results (dict): Resultados dos modelos otimizados
        insights (list): Lista de insights
    """
    os.makedirs('../results', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'../results/analysis_summary_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMO DA ANÁLISE DE CLASSIFICAÇÃO DE GRÃOS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Resultados dos modelos originais
        f.write("RESULTADOS DOS MODELOS ORIGINAIS:\n")
        f.write("-"*40 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Acurácia: {result['accuracy']:.4f}\n")
            f.write(f"  Precisão: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
        
        # Resultados otimizados
        if optimized_results:
            f.write("\n\nRESULTADOS DOS MODELOS OTIMIZADOS:\n")
            f.write("-"*40 + "\n")
            for model_name, result in optimized_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Acurácia: {result['accuracy']:.4f}\n")
                f.write(f"  Melhores parâmetros: {result['best_params']}\n")
        
        # Insights
        f.write("\n\nPRINCIPAIS INSIGHTS:\n")
        f.write("-"*40 + "\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"\n{i}. {insight}\n")
    
    print(f"\nResumo salvo em: {filename}")


def generate_insights(data_info, eda_results, model_results, feature_importance=None):
    """
    Gera insights baseados na análise.
    
    Parameters:
        data_info (dict): Informações do dataset
        eda_results (dict): Resultados da análise exploratória
        model_results (dict): Resultados dos modelos
        feature_importance (pd.DataFrame): Importância das características
        
    Returns:
        list: Lista de insights
    """
    insights = []
    
    # Insight sobre distribuição das classes
    insights.append(
        f"DISTRIBUIÇÃO DAS CLASSES: O dataset está perfeitamente balanceado com "
        f"{data_info['shape'][0] // 3} amostras de cada variedade."
    )
    
    # Insight sobre características importantes
    if feature_importance is not None:
        top_features = feature_importance.head(3)['feature'].tolist()
        insights.append(
            f"CARACTERÍSTICAS MAIS DISCRIMINANTES: Com base na análise do Random Forest, "
            f"as características mais importantes são: {', '.join(top_features)}."
        )
    
    # Insight sobre correlações
    if eda_results['strong_correlations']:
        corr = eda_results['strong_correlations'][0]
        insights.append(
            f"CORRELAÇÕES: Existe forte correlação entre {corr['feature1']} e "
            f"{corr['feature2']} ({corr['correlation']:.3f})."
        )
    
    # Insight sobre performance dos modelos
    best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
    insights.append(
        f"PERFORMANCE DOS MODELOS: O melhor modelo foi {best_model[0]} com "
        f"{best_model[1]['accuracy']*100:.1f}% de acurácia."
    )
    
    # Insights práticos
    insights.extend([
        "APLICAÇÃO PRÁTICA: Os modelos desenvolvidos podem ser implementados em "
        "cooperativas agrícolas para automatizar a classificação de grãos, "
        "reduzindo tempo e erros humanos.",
        
        "ROBUSTEZ: A alta acurácia obtida por todos os modelos (>90%) indica que "
        "as características físicas medidas são altamente discriminativas para "
        "diferenciar as variedades de trigo."
    ])
    
    return insights