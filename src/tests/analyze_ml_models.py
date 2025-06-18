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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_loader import load_seeds_data
from config import FEATURE_NAMES, VARIETY_NAMES

def analyze_ml_models():
    """Analisa detalhadamente os modelos de ML."""
    
    print("="*80)
    print("ANÁLISE DETALHADA DOS MODELOS DE MACHINE LEARNING")
    print("="*80)
    
    # Carregar dados
    data = load_seeds_data()
    X = data[FEATURE_NAMES]
    y = data['variety']
    
    # Normalização
    print("\n1. PREPROCESSAMENTO DOS DADOS")
    print("="*50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Normalização Z-score aplicada:")
    for i, feature in enumerate(FEATURE_NAMES):
        original_mean = X[feature].mean()
        original_std = X[feature].std()
        scaled_mean = X_scaled[:, i].mean()
        scaled_std = X_scaled[:, i].std()
        print(f"  {feature}:")
        print(f"    Original: μ={original_mean:.3f}, σ={original_std:.3f}")
        print(f"    Normalizado: μ={scaled_mean:.1e}, σ={scaled_std:.3f}")
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDivisão dos dados:")
    print(f"  Total de amostras: {len(data)}")
    print(f"  Conjunto de treino: {len(X_train)} amostras ({len(X_train)/len(data)*100:.1f}%)")
    print(f"  Conjunto de teste: {len(X_test)} amostras ({len(X_test)/len(data)*100:.1f}%)")
    
    print("\nDistribuição no conjunto de teste:")
    for variety in sorted(y_test.unique()):
        count = sum(y_test == variety)
        print(f"  Variedade {variety}: {count} amostras ({count/len(y_test)*100:.1f}%)")
    
    # Modelos base
    print("\n2. MODELOS BASE (SEM OTIMIZAÇÃO)")
    print("="*50)
    
    base_models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Naive Bayes': GaussianNB()
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
    
    # KNN
    print("\nKNN - Grid Search:")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    
    print(f"  Total de combinações testadas: {len(knn_grid.cv_results_['params'])}")
    print(f"  Melhor configuração: {knn_grid.best_params_}")
    print(f"  Score CV: {knn_grid.best_score_:.4f}")
    
    # SVM
    print("\nSVM - Grid Search:")
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    
    print(f"  Total de combinações testadas: {len(svm_grid.cv_results_['params'])}")
    print(f"  Melhor configuração: {svm_grid.best_params_}")
    print(f"  Score CV: {svm_grid.best_score_:.4f}")
    
    # Random Forest
    print("\nRandom Forest - Grid Search:")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    print(f"  Total de combinações testadas: {len(rf_grid.cv_results_['params'])}")
    print(f"  Melhor configuração: {rf_grid.best_params_}")
    print(f"  Score CV: {rf_grid.best_score_:.4f}")
    
    # Modelos otimizados
    print("\n4. RESULTADOS DOS MODELOS OTIMIZADOS")
    print("="*50)
    
    optimized_models = {
        'KNN': knn_grid.best_estimator_,
        'SVM': svm_grid.best_estimator_,
        'Random Forest': rf_grid.best_estimator_,
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
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
            print("  Predito →")
            print("Real ↓    1    2    3")
            for i in range(3):
                row = f"  {i+1}     "
                for j in range(3):
                    row += f"{cm[i,j]:3d}  "
                print(row)
            
            # Análise de erros
            total_errors = len(y_test) - sum(cm[i,i] for i in range(3))
            print(f"\nAnálise de erros:")
            print(f"  Total de erros: {total_errors}")
            for i in range(3):
                for j in range(3):
                    if i != j and cm[i,j] > 0:
                        # Mapear índices de classe para nomes de variedade
                        variety_names_map = {0: 'Kama', 1: 'Rosa', 2: 'Canadian'}
                        print(f"  {variety_names_map[i]} → {variety_names_map[j]}: {cm[i,j]} erro(s)")
        
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
    
    rf_model = optimized_models['Random Forest']
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
    
    for name, results in sorted_models:
        cv_str = f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
        cv_coef = (results['cv_std'] / results['cv_mean']) * 100
        
        if cv_coef < 5:
            stability = "Excelente"
        elif cv_coef < 10:
            stability = "Boa"
        else:
            stability = "Moderada"
        
        print(f"{name:<20} {results['accuracy']:.4f}     {cv_str:<15} {stability}")
    
    return detailed_results

if __name__ == "__main__":
    analyze_ml_models()