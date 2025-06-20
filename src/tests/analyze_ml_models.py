#!/usr/bin/env python3
"""Script para analisar detalhadamente os modelos de ML e seus resultados."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, f1_score
from data_loader import load_seeds_data
from config import FEATURE_NAMES, VARIETY_NAMES, RANDOM_SEED, TEST_SIZE, CV_FOLDS, set_random_seeds
from utils import save_test_output, save_test_results_json, create_directories

def analyze_ml_models():
    """Analisa detalhadamente os modelos de ML com reprodutibilidade garantida."""
    
    # Configurar diretórios de output para testes
    create_directories(is_test=True, subdirs=['ml_models'])
    
    # CRÍTICO: Configurar reprodutibilidade ANTES de qualquer operação
    set_random_seeds()
    
    print("="*80)
    print("ANÁLISE DETALHADA DOS MODELOS DE MACHINE LEARNING")
    print("="*80)
    print(f"🔧 Configuração de reprodutibilidade: RANDOM_SEED = {RANDOM_SEED}")
    
    # Carregar dados
    data = load_seeds_data()
    
    # Análise básica usando pandas
    print(f"Dataset shape: {data.shape}")
    print(f"Variedades disponíveis: {list(VARIETY_NAMES.values())}")
    print(f"Códigos das variedades: {list(VARIETY_NAMES.keys())}")
    
    # Distribuição de classes usando pandas
    class_distribution = data['variety'].value_counts().sort_index()
    print(f"Distribuição por variedade:")
    for variety_code, count in class_distribution.items():
        variety_name = VARIETY_NAMES[variety_code]
        percentage = (count / len(data)) * 100
        print(f"  {variety_name} (código {variety_code}): {count} amostras ({percentage:.1f}%)")
    
    X = data[FEATURE_NAMES]
    y = data['variety']
    
    # Normalização
    print("\n1. PREPROCESSAMENTO DOS DADOS")
    print("="*50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Normalização Z-score aplicada:")
    
    # Usar pandas para estatísticas mais elegantes
    original_stats = X.describe().loc[['mean', 'std']].round(3)
    
    for i, feature in enumerate(FEATURE_NAMES):
        original_mean = original_stats.loc['mean', feature]
        original_std = original_stats.loc['std', feature]
        scaled_mean = X_scaled[:, i].mean()
        scaled_std = X_scaled[:, i].std()
        print(f"  {feature}:")
        print(f"    Original: μ={original_mean:.3f}, σ={original_std:.3f}")
        print(f"    Normalizado: μ={scaled_mean:.1e}, σ={scaled_std:.3f}")
    
    # Divisão treino/teste com configuração centralizada
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"\nDivisão dos dados:")
    print(f"  Total de amostras: {len(data)}")
    print(f"  Conjunto de treino: {len(X_train)} amostras ({len(X_train)/len(data)*100:.1f}%)")
    print(f"  Conjunto de teste: {len(X_test)} amostras ({len(X_test)/len(data)*100:.1f}%)")
    
    print("\nDistribuição no conjunto de teste:")
    for variety_code in sorted(y_test.unique()):
        variety_name = VARIETY_NAMES[variety_code]
        count = sum(y_test == variety_code)
        print(f"  {variety_name} (código {variety_code}): {count} amostras ({count/len(y_test)*100:.1f}%)")
    
    # Modelos base
    print("\n2. MODELOS BASE (SEM OTIMIZAÇÃO)")
    print("="*50)
    
    # Modelos base EXATAMENTE como na documentação (seção 4.1)
    base_models = {
        'KNN': KNeighborsClassifier(
            n_neighbors=5,      # Valor padrão k=5 para balancear bias-variance
            metric='euclidean', # Métrica euclidiana inicial
            weights='uniform'   # Peso uniforme para vizinhos
        ),
        'SVM': SVC(
            kernel='rbf',       # Kernel RBF para não-linearidade inicial
            C=1.0,              # Regularização padrão
            gamma='scale',      # Escala automática do kernel
            random_state=RANDOM_SEED     # Reprodutibilidade
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,   # 100 árvores para estabilidade
            max_depth=None,     # Profundidade ilimitada inicialmente
            min_samples_split=2,# Critério padrão de divisão
            random_state=RANDOM_SEED     # Reprodutibilidade
        ),
        'LogisticRegression': LogisticRegression(
            penalty='l2',       # Regularização Ridge
            C=1.0,              # Força de regularização padrão
            solver='lbfgs',     # Otimizador quasi-Newton
            max_iter=1000,      # Iterações suficientes para convergência
            random_state=RANDOM_SEED     # Reprodutibilidade
        ),
        'NaiveBayes': GaussianNB(
            var_smoothing=1e-9  # Suavização para estabilidade numérica
        )
    }
    
    base_results = {}
    
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        base_results[name] = acc
        print(f"{name}: {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)} corretas)")
    
    # Otimização de hiperparâmetros
    print("\n3. OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("="*50)
    
    # Espaços de busca EXATOS da documentação (seção 4.2)
    param_grids = {
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],  # 6 valores: ímpar para evitar empates
            'metric': ['euclidean', 'manhattan', 'minkowski'],  # 3 métricas de distância
            'weights': ['uniform', 'distance']  # 2 esquemas de ponderação
            # Total: 6 × 3 × 2 = 36 combinações
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],  # 4 valores de regularização (escala logarítmica)
            'kernel': ['linear', 'rbf', 'poly'],  # 3 tipos de kernel
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # 6 valores para kernels não-lineares
            # Total: 4 × 3 × 6 = 72 combinações
        },
        'RandomForest': {
            'n_estimators': [10, 50, 100, 200],  # 4 valores: número de árvores
            'max_depth': [None, 5, 10, 20],  # 4 valores: profundidade máxima
            'min_samples_split': [2, 5, 10],  # 3 valores: amostras mínimas para divisão
            'min_samples_leaf': [1, 2, 4]  # 3 valores: amostras mínimas em folhas
            # Total: 4 × 4 × 3 × 3 = 144 combinações
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],  # 5 valores de regularização
            'penalty': ['l2'],  # L2 para compatibilidade
            'solver': ['lbfgs']  # Solver moderno recomendado
            # Total: 5 × 1 × 1 = 5 combinações
        },
        'NaiveBayes': {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]  # 4 valores de suavização
            # Total: 4 combinações
        }
    }
    
    optimization_results = {}
    
    for model_name in ['KNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes']:
        print(f"\n{model_name} - Grid Search:")
        
        # Modelo base para otimização
        if model_name == 'KNN':
            base_model = KNeighborsClassifier()
        elif model_name == 'SVM':
            base_model = SVC(random_state=RANDOM_SEED)
        elif model_name == 'RandomForest':
            base_model = RandomForestClassifier(random_state=RANDOM_SEED)
        elif model_name == 'LogisticRegression':
            base_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        elif model_name == 'NaiveBayes':
            base_model = GaussianNB()
        
        param_grid = param_grids[model_name]
        
        # Grid search com CV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=CV_FOLDS,
            scoring='accuracy',
            n_jobs=1,  # Para garantir reprodutibilidade
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        optimization_results[model_name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'total_combinations': len(grid_search.cv_results_['params'])
        }
        
        print(f"  Total de combinações testadas: {optimization_results[model_name]['total_combinations']}")
        print(f"  Melhor configuração: {optimization_results[model_name]['best_params']}")
        print(f"  Score CV: {optimization_results[model_name]['best_score']:.4f}")
    
    # Modelos otimizados (usando nomes consistentes com documentação)
    print("\n4. RESULTADOS DOS MODELOS OTIMIZADOS")
    print("="*50)
    
    optimized_models = {}
    for model_name in ['KNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes']:
        if model_name in optimization_results:
            optimized_models[model_name] = optimization_results[model_name]['best_estimator']
        else:
            # Fallback para modelos base se não otimizados
            optimized_models[model_name] = base_models[model_name]
    
    detailed_results = {}
    
    for name, model in optimized_models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Treinar e prever
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Cross-validation detalhada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"Acurácia no teste: {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)} corretas)")
        print(f"Precisão (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        
        print(f"\nCross-validation 5-fold:")
        print(f"  Scores individuais: {[round(s, 4) for s in cv_scores]}")
        print(f"  Média: {cv_scores.mean():.4f}")
        print(f"  Desvio padrão: {cv_scores.std():.4f}")
        print(f"  Coeficiente de variação: {(cv_scores.std()/cv_scores.mean())*100:.2f}%")
        
        # Melhoria com otimização
        if name in base_results:
            improvement = (acc - base_results[name]) / base_results[name] * 100
            print(f"\nMelhoria com otimização: {improvement:+.2f}%")
        
        # Matriz de confusão para os melhores modelos
        if name in ['KNN', 'SVM']:
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nMatriz de confusão:")
            # Criar mapeamento usando VARIETY_NAMES
            sorted_variety_codes = sorted(y_test.unique())
            variety_names = [VARIETY_NAMES[code] for code in sorted_variety_codes]
            
            print("  Predito →")
            header = "Real ↓    "
            for name in variety_names:
                header += f"{name[:4]:>5}"  # Primeiras 4 letras, alinhadas à direita
            print(header)
            
            for i, real_name in enumerate(variety_names):
                row = f"  {real_name[:4]:<4}   "  # Nome real à esquerda
                for j in range(3):
                    row += f"{cm[i,j]:4d} "
                print(row)
            
            # Análise de erros usando VARIETY_NAMES do config
            total_errors = len(y_test) - sum(cm[i,i] for i in range(3))
            print(f"\nAnálise de erros:")
            print(f"  Total de erros: {total_errors}")
            
            # Criar mapeamento correto usando VARIETY_NAMES
            sorted_variety_codes = sorted(y_test.unique())
            
            for i in range(3):
                for j in range(3):
                    if i != j and cm[i,j] > 0:
                        variety_code_i = sorted_variety_codes[i]
                        variety_code_j = sorted_variety_codes[j]
                        variety_name_i = VARIETY_NAMES[variety_code_i]
                        variety_name_j = VARIETY_NAMES[variety_code_j]
                        print(f"  {variety_name_i} → {variety_name_j}: {cm[i,j]} erro(s)")
        
        detailed_results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Feature importance (Random Forest)
    print("\n5. IMPORTÂNCIA DAS FEATURES (RANDOM FOREST)")
    print("="*50)
    
    rf_model = optimized_models['RandomForest']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Ranking de importância:")
    cumulative = 0
    for i, idx in enumerate(indices):
        importance = importances[idx]
        cumulative += importance
        print(f"{i+1}. {FEATURE_NAMES[idx]}: {importance:.3f} ({cumulative*100:.1f}% acumulado)")
    
    # Resumo comparativo
    print("\n6. RESUMO COMPARATIVO")
    print("="*50)
    
    print("\nTabela de resultados:")
    print(f"{'Modelo':<20} {'Acurácia':<10} {'CV Score':<15} {'Estabilidade'}")
    print("-" * 60)
    
    sorted_models = sorted(detailed_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Criar DataFrame pandas para resultados mais estruturados
    results_data = []
    for name, results in sorted_models:
        cv_str = f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
        cv_coef = (results['cv_std'] / results['cv_mean']) * 100
        
        if cv_coef < 5:
            stability = "Excelente"
        elif cv_coef < 10:
            stability = "Boa"
        else:
            stability = "Moderada"
        
        results_data.append({
            'Modelo': name,
            'Acurácia': results['accuracy'],
            'CV_Mean': results['cv_mean'],
            'CV_Std': results['cv_std'],
            'Estabilidade': stability
        })
        
        print(f"{name:<20} {results['accuracy']:.4f}     {cv_str:<15} {stability}")
    
    # Salvar resultados em DataFrame para possível análise posterior
    results_df = pd.DataFrame(results_data)
    print(f"\n📊 Resultados salvos em DataFrame com {len(results_df)} modelos")
    
    # Preparar dados estruturados para salvamento
    structured_results = {
        'config': {
            'random_seed': RANDOM_SEED,
            'test_size': TEST_SIZE,
            'cv_folds': CV_FOLDS,
            'dataset_shape': data.shape
        },
        'models': detailed_results,
        'summary': {
            'best_model': max(detailed_results.items(), key=lambda x: x[1]['accuracy'])[0],
            'best_accuracy': max(detailed_results.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy'],
            'total_models': len(detailed_results)
        }
    }
    
    # Salvar resultados estruturados em JSON
    save_test_results_json(structured_results, 'ml_models', 'detailed_analysis.json')
    
    # Preparar summary textual
    summary_text = f"""
RESUMO DA ANÁLISE ML
{'='*50}

Configuração:
- Random Seed: {RANDOM_SEED}
- Test Size: {TEST_SIZE}
- CV Folds: {CV_FOLDS}
- Dataset: {data.shape[0]} amostras, {data.shape[1]} features

Modelos Analisados: {len(detailed_results)}

Melhores Resultados:
"""
    
    for name, results in sorted(detailed_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        summary_text += f"\n{name}:"
        summary_text += f"\n  - Acurácia: {results['accuracy']:.4f}"
        summary_text += f"\n  - CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
        summary_text += f"\n  - Precisão: {results['precision']:.4f}"
        summary_text += f"\n  - F1-Score: {results['f1']:.4f}"
    
    # Salvar summary em texto
    save_test_output(summary_text, 'ml_models', 'summary.txt')
    
    print(f"\n✅ Todos os resultados salvos em test_results/ml_models/")
    
    return detailed_results

def generate_documentation_data():
    """Gera dados específicos para atualização da documentação (seção 4.2 e 4.3)."""
    
    # Configurar reprodutibilidade
    set_random_seeds()
    
    print("="*80)
    print("GERAÇÃO DE DADOS PARA DOCUMENTAÇÃO (SEÇÕES 4.2 e 4.3)")
    print("="*80)
    print(f"🔧 RANDOM_SEED = {RANDOM_SEED}, TEST_SIZE = {TEST_SIZE}, CV_FOLDS = {CV_FOLDS}")
    
    # Carregar e preparar dados
    data = load_seeds_data()
    
    print(f"Variedades: {list(VARIETY_NAMES.values())}")
    
    X = data[FEATURE_NAMES]
    y = data['variety']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"\nDados: {len(data)} amostras → {len(X_train)} treino + {len(X_test)} teste")
    
    # Modelos base EXATAMENTE como na documentação (seção 4.1)
    models = {
        'KNN': KNeighborsClassifier(
            n_neighbors=5,      # Valor padrão k=5 para balancear bias-variance
            metric='euclidean', # Métrica euclidiana inicial
            weights='uniform'   # Peso uniforme para vizinhos
        ),
        'SVM': SVC(
            kernel='rbf',       # Kernel RBF para não-linearidade inicial
            C=1.0,              # Regularização padrão
            gamma='scale',      # Escala automática do kernel
            random_state=RANDOM_SEED     # Reprodutibilidade
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,   # 100 árvores para estabilidade
            max_depth=None,     # Profundidade ilimitada inicialmente
            min_samples_split=2,# Critério padrão de divisão
            random_state=RANDOM_SEED     # Reprodutibilidade
        ),
        'NaiveBayes': GaussianNB(
            var_smoothing=1e-9  # Suavização para estabilidade numérica
        ),
        'LogisticRegression': LogisticRegression(
            penalty='l2',       # Regularização Ridge
            C=1.0,              # Força de regularização padrão
            solver='lbfgs',     # Otimizador quasi-Newton
            max_iter=1000,      # Iterações suficientes para convergência
            random_state=RANDOM_SEED     # Reprodutibilidade
        )
    }
    
    # Espaços de busca EXATOS da seção 4.2
    param_grids = {
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'weights': ['uniform', 'distance']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        'RandomForest': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        'NaiveBayes': {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]
        }
    }
    
    print("\n📊 OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("="*60)
    
    optimization_results = {}
    
    for model_name in ['KNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes']:
        model = models[model_name]
        param_grid = param_grids[model_name]
        
        print(f"\n🔍 Otimizando {model_name}...")
        
        # Baseline score
        baseline_cv = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
        baseline_score = baseline_cv.mean()
        
        # Grid search com reprodutibilidade
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=CV_FOLDS,
            scoring='accuracy',
            n_jobs=1,  # Para garantir reprodutibilidade
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        optimized_score = grid_search.best_score_
        improvement = optimized_score - baseline_score
        improvement_pct = (improvement / baseline_score) * 100
        
        optimization_results[model_name] = {
            'baseline': baseline_score,
            'optimized': optimized_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'best_params': grid_search.best_params_,
            'combinations': len(grid_search.cv_results_['params'])
        }
        
        print(f"  ✅ Baseline CV: {baseline_score:.4f} ({baseline_score*100:.2f}%)")
        print(f"  ✅ Otimizado CV: {optimized_score:.4f} ({optimized_score*100:.2f}%)")
        print(f"  ✅ Melhoria: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print(f"  ✅ Combinações: {optimization_results[model_name]['combinations']}")
        print(f"  ✅ Melhor config: {grid_search.best_params_}")
    
    print("\n📋 TABELA PARA SEÇÃO 4.2")
    print("="*60)
    print("| Algorithm | Search Space | Best Config | Improvement |")
    print("|-----------|--------------|-------------|-------------|")
    
    for model_name in ['KNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes']:
        result = optimization_results[model_name]
        combinations = result['combinations']
        best_params = result['best_params']
        
        # Formatação da configuração para a tabela
        if model_name == 'KNN':
            config = f"n_neighbors={best_params['n_neighbors']}, metric='{best_params['metric']}', weights='{best_params['weights']}'"
        elif model_name == 'SVM':
            if 'gamma' in best_params:
                config = f"C={best_params['C']}, kernel='{best_params['kernel']}', gamma='{best_params['gamma']}'"
            else:
                config = f"C={best_params['C']}, kernel='{best_params['kernel']}'"
        elif model_name == 'RandomForest':
            config = f"n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}"
        elif model_name == 'LogisticRegression':
            config = f"C={best_params['C']}, penalty='{best_params['penalty']}', solver='{best_params['solver']}'"
        elif model_name == 'NaiveBayes':
            config = f"var_smoothing={best_params['var_smoothing']}"
        
        improvement = result['improvement_pct']
        if improvement >= 0:
            improvement_str = f'+{improvement:.2f}%'
        else:
            improvement_str = f'{improvement:.2f}%'
            if improvement < -1:
                improvement_str += '*'
        
        print(f"| {model_name} | {combinations} combinations | `{config}` | {improvement_str} |")
    
    print("\n📊 PERFORMANCE NO CONJUNTO DE TESTE")
    print("="*60)
    
    # Modelos otimizados para teste
    test_results = {}
    all_models = ['KNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes']
    
    for model_name in all_models:
        if model_name in optimization_results:
            # Usar parâmetros otimizados
            best_params = optimization_results[model_name]['best_params']
            if model_name == 'KNN':
                optimized_model = KNeighborsClassifier(**best_params)
            elif model_name == 'SVM':
                optimized_model = SVC(random_state=RANDOM_SEED, **best_params)
            elif model_name == 'RandomForest':
                optimized_model = RandomForestClassifier(random_state=RANDOM_SEED, **best_params)
            elif model_name == 'LogisticRegression':
                optimized_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, **best_params)
            elif model_name == 'NaiveBayes':
                optimized_model = GaussianNB(**best_params)
        else:
            # Usar modelo padrão
            optimized_model = models[model_name]
        
        # Treinar e avaliar
        optimized_model.fit(X_train, y_train)
        y_pred = optimized_model.predict(X_test)
        
        # Métricas de teste
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation no treino
        cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        test_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'correct_predictions': int(round(accuracy * len(y_test))),
            'total_samples': len(y_test)
        }
        
        print(f"\n🎯 {model_name}:")
        print(f"  Acurácia teste: {accuracy:.4f} ({test_results[model_name]['correct_predictions']}/{len(y_test)})")
        print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    
    print("\n📋 TABELA PARA SEÇÃO 4.3")
    print("="*60)
    print("| Algorithm | Accuracy | Precision | Recall | F1-Score | CV Score |")
    print("|-----------|----------|-----------|---------|----------|----------|")
    
    for model_name in all_models:
        if model_name in test_results:
            result = test_results[model_name]
            acc = result['accuracy']
            prec = result['precision']
            rec = result['recall']
            f1 = result['f1']
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']
            
            # Destacar os melhores (threshold ajustável)
            if acc >= 0.888:
                prefix = '**'
                suffix = '**'
            else:
                prefix = ''
                suffix = ''
            
            print(f"| {prefix}{model_name}{suffix} | {prefix}{acc*100:.2f}%{suffix} | {prec*100:.2f}% | {rec*100:.2f}% | {f1*100:.2f}% | {cv_mean*100:.2f}% ± {cv_std*100:.2f}% |")
    
    # Identificar melhor modelo
    best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['accuracy'])
    best_result = test_results[best_model_name]
    
    print(f"\n🏆 MELHOR MODELO IDENTIFICADO")
    print("="*60)
    print(f"Modelo: {best_model_name}")
    print(f"Acurácia: {best_result['accuracy']:.4f} = {best_result['accuracy']*100:.2f}%")
    print(f"Predições corretas: {best_result['correct_predictions']}/{best_result['total_samples']}")
    
    if best_model_name in optimization_results:
        print(f"Configuração otimizada: {optimization_results[best_model_name]['best_params']}")
    
    print(f"\n📝 DADOS PARA CÓDIGO DE EXEMPLO NA SEÇÃO 4.3")
    print("="*60)
    print(f"total_test_samples = {best_result['total_samples']}  # amostras de teste")
    print(f"correct_predictions = {best_result['correct_predictions']}  # predições corretas")
    print(f"accuracy_manual = {best_result['correct_predictions']}/{best_result['total_samples']} = {best_result['accuracy']:.4f}")
    print(f"# Output: {best_result['accuracy']*100:.2f}%")
    
    return optimization_results, test_results

if __name__ == "__main__":
    # Executar análise completa
    detailed_results = analyze_ml_models()
    
    print("\n" + "="*80)
    print("EXECUTANDO GERAÇÃO ESPECÍFICA PARA DOCUMENTAÇÃO")
    print("="*80)
    
    # Executar geração de dados para documentação
    opt_results, test_results = generate_documentation_data()