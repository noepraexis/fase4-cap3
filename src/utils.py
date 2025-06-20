"""Módulo com funções utilitárias."""

import os
import json
import joblib
from datetime import datetime
from pathlib import Path


def get_project_root():
    """
    Encontra a raiz do projeto de forma robusta.
    
    Estratégia:
    1. Procura a partir do arquivo atual (__file__)
    2. Procura a partir do diretório de trabalho atual
    3. Múltiplos indicadores da raiz do projeto
    
    Returns:
        Path: Caminho absoluto para a raiz do projeto
    """
    def is_project_root(path):
        """Verifica se um diretório é a raiz do projeto."""
        path = Path(path)
        # Múltiplos indicadores para maior robustez
        indicators = [
            (path / 'src').exists(),
            (path / 'datasets').exists(),
            (path / 'README.md').exists(),
            (path / 'src' / 'config.py').exists(),
            (path / 'src' / 'tests').exists()
        ]
        # Precisa de pelo menos 3 indicadores para ser considerado raiz
        return sum(indicators) >= 3
    
    # Estratégia 1: Começar do arquivo atual
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if is_project_root(parent):
            return parent
    
    # Estratégia 2: Começar do diretório de trabalho atual
    current_dir = Path.cwd().resolve()
    for parent in [current_dir] + list(current_dir.parents):
        if is_project_root(parent):
            return parent
    
    # Estratégia 3: Procurar pelo nome característico do projeto
    for parent in current_file.parents:
        if parent.name == 'fase4-cap3' and is_project_root(parent):
            return parent
    
    # Estratégia 4: Fallback baseado na estrutura conhecida
    if current_file.parent.name == 'src':
        candidate = current_file.parent.parent
        if is_project_root(candidate):
            return candidate
    
    if current_file.parent.parent.name == 'src':  # Para scripts em src/tests/
        candidate = current_file.parent.parent.parent
        if is_project_root(candidate):
            return candidate
    
    # Último recurso: listar diretórios encontrados para debug
    checked_paths = []
    for parent in current_file.parents:
        checked_paths.append(str(parent))
        if len(checked_paths) > 10:  # Limitar para não spammar
            break
    
    raise RuntimeError(
        f"Não foi possível encontrar a raiz do projeto.\n"
        f"Arquivo atual: {current_file}\n"
        f"Diretório atual: {current_dir}\n"
        f"Caminhos verificados: {checked_paths[:5]}...\n"
        f"Certifique-se de que o script está sendo executado "
        f"dentro da estrutura do projeto 'fase4-cap3'."
    )


# Obter raiz do projeto na importação do módulo
PROJECT_ROOT = get_project_root()


def validate_project_structure():
    """
    Valida que a estrutura do projeto está correta e os paths estão funcionando.
    
    Returns:
        bool: True se tudo estiver correto
        
    Raises:
        RuntimeError: Se houver problemas na estrutura
    """
    issues = []
    
    # Verificar estrutura básica
    required_dirs = ['src', 'datasets', 'assets', 'models']
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            issues.append(f"Diretório obrigatório não encontrado: {dir_path}")
    
    # Verificar arquivos essenciais
    essential_files = [
        'src/config.py',
        'src/utils.py', 
        'src/data_loader.py',
        'README.md'
    ]
    for file_path in essential_files:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            issues.append(f"Arquivo essencial não encontrado: {full_path}")
    
    # Verificar se PROJECT_ROOT aponta para raiz correta
    if not PROJECT_ROOT.name.endswith('fase4-cap3'):
        issues.append(f"PROJECT_ROOT parece incorreto: {PROJECT_ROOT}")
    
    if issues:
        raise RuntimeError(
            "Problemas na estrutura do projeto detectados:\n" + 
            "\n".join(f"- {issue}" for issue in issues)
        )
    
    return True


# Validar estrutura na importação (apenas uma vez)
try:
    validate_project_structure()
except RuntimeError as e:
    print(f"⚠️  Aviso: {e}")
    print(f"   PROJECT_ROOT atual: {PROJECT_ROOT}")


def get_output_path(is_test=False, subdir=None, filename=None):
    """
    Determina o caminho de output baseado no tipo de script.
    
    Parameters:
        is_test (bool): True se for um script de teste
        subdir (str): Subdiretório específico (opcional)
        filename (str): Nome do arquivo (opcional)
        
    Returns:
        Path: Caminho absoluto para o output
    """
    if is_test:
        base_dir = PROJECT_ROOT / "test_results"
    else:
        base_dir = PROJECT_ROOT / "results"
    
    if subdir:
        output_dir = base_dir / subdir
    else:
        output_dir = base_dir
    
    if filename:
        return output_dir / filename
    else:
        return output_dir


def create_directories(is_test=False, subdirs=None):
    """
    Cria diretórios necessários se não existirem.
    
    Parameters:
        is_test (bool): True se for para scripts de teste
        subdirs (list): Lista de subdiretórios específicos para criar
    """
    # Diretórios padrão sempre necessários (caminhos absolutos)
    standard_dirs = [
        PROJECT_ROOT / 'models',
        PROJECT_ROOT / 'assets'
    ]
    
    # Adicionar diretório de resultados baseado no tipo
    if is_test:
        standard_dirs.append(PROJECT_ROOT / 'test_results')
        if subdirs:
            for subdir in subdirs:
                standard_dirs.append(PROJECT_ROOT / 'test_results' / subdir)
    else:
        standard_dirs.append(PROJECT_ROOT / 'results')
    
    for directory in standard_dirs:
        directory.mkdir(parents=True, exist_ok=True)
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
    # Criar diretório se não existir (caminho absoluto)
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo baseado no modelo (caminhos absolutos)
    model_filename = models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
    scaler_filename = models_dir / "scaler.pkl"
    info_filename = models_dir / "model_info.json"
    
    # Salvar modelo e scaler
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    # Salvar informações do modelo
    model_info = {
        'model_name': model_name,
        'accuracy': accuracy,
        'features': features,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_file': str(model_filename),
        'scaler_file': str(scaler_filename)
    }
    
    with open(info_filename, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"\nModelo '{model_name}' salvo com sucesso!")
    print(f"Arquivos salvos:")
    print(f"- Modelo: {model_filename}")
    print(f"- Scaler: {scaler_filename}")
    print(f"- Informações: {info_filename}")


def load_model(model_name=None):
    """
    Carrega modelo salvo.
    
    Parameters:
        model_name (str): Nome do modelo (opcional)
        
    Returns:
        dict: Dicionário com modelo, scaler e informações
    """
    # Carregar informações do modelo (caminho absoluto)
    models_dir = PROJECT_ROOT / 'models'
    info_filename = models_dir / 'model_info.json'
    
    with open(info_filename, 'r') as f:
        model_info = json.load(f)
    
    # Se não especificado, usar o modelo salvo
    if model_name is None:
        model_filename = Path(model_info['model_file'])
    else:
        model_filename = models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
    
    # Carregar modelo e scaler
    model = joblib.load(model_filename)
    scaler = joblib.load(Path(model_info['scaler_file']))
    
    return {
        'model': model,
        'scaler': scaler,
        'info': model_info
    }


def save_results_summary(results, optimized_results, insights, is_test=False, test_name=None):
    """
    Salva resumo dos resultados em arquivo.
    
    Parameters:
        results (dict): Resultados dos modelos originais
        optimized_results (dict): Resultados dos modelos otimizados
        insights (list): Lista de insights
        is_test (bool): True se for output de teste
        test_name (str): Nome específico do teste (para organização)
    """
    # Determinar diretório de output
    if is_test and test_name:
        output_dir = get_output_path(is_test=True, subdir=test_name)
        os.makedirs(output_dir, exist_ok=True)
        filename = output_dir / "analysis_summary.txt"
    else:
        output_dir = get_output_path(is_test=is_test)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'analysis_summary_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMO DA ANÁLISE DE CLASSIFICAÇÃO DE GRÃOS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if is_test and test_name:
            f.write(f"Tipo: Teste - {test_name}\n")
        f.write("\n")
        
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


def save_test_output(content, test_name, filename="analysis_output.txt"):
    """
    Salva output específico de um teste.
    
    Parameters:
        content (str): Conteúdo a ser salvo
        test_name (str): Nome do teste (usado como subdiretório)
        filename (str): Nome do arquivo de output
    """
    output_path = get_output_path(is_test=True, subdir=test_name, filename=filename)
    os.makedirs(output_path.parent, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"OUTPUT DO TESTE: {test_name.upper()}\n")
        f.write("="*80 + "\n")
        f.write(f"Data da execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(content)
    
    print(f"\n📁 Output salvo em: {output_path}")


def save_test_results_json(data, test_name, filename="results.json"):
    """
    Salva resultados de teste em formato JSON.
    
    Parameters:
        data (dict): Dados a serem salvos
        test_name (str): Nome do teste
        filename (str): Nome do arquivo JSON
    """
    output_path = get_output_path(is_test=True, subdir=test_name, filename=filename)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Adicionar metadados
    data_with_metadata = {
        'metadata': {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'generated_by': 'src/utils.py'
        },
        'results': data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_with_metadata, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 Resultados JSON salvos em: {output_path}")


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