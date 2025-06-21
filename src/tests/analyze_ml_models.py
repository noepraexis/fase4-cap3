#!/usr/bin/env python3
"""Análise completa de modelos de Machine Learning para classificação de grãos."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from base_script import MLScriptBase
from config import FEATURE_NAMES, VARIETY_NAMES, CV_FOLDS


class MLModelsAnalyzer(MLScriptBase):
    """Analisador completo de modelos de Machine Learning."""
    
    def __init__(self):
        super().__init__("ml_models")
        self.models_config = self._get_models_config()
        self.param_grids = self._get_param_grids()
    
    def run_complete_analysis(self):
        """Executa análise completa dos modelos."""
        # 1. Carregar e preparar dados
        data = self.load_data()
        X_train, X_test, y_train, y_test, scaler = self.split_data()
        
        # 2. Análise dos modelos base
        print("\n📊 ANÁLISE DE MODELOS BASE")
        print("=" * 50)
        base_results = self._analyze_base_models(X_train, X_test, y_train, y_test)
        
        # 3. Otimização de hiperparâmetros  
        print("\n🔧 OTIMIZAÇÃO DE HIPERPARÂMETROS")
        print("=" * 50)
        optimization_results = self._optimize_models(X_train, y_train)
        
        # 4. Avaliação completa dos modelos otimizados
        print("\n📈 AVALIAÇÃO DOS MODELOS OTIMIZADOS")
        print("=" * 50)
        final_results = self._evaluate_optimized_models(
            X_train, X_test, y_train, y_test, optimization_results
        )
        
        # 5. Análise de importância das features
        print("\n🎯 IMPORTÂNCIA DAS FEATURES")
        print("=" * 50)
        feature_importance = self._analyze_feature_importance(final_results)
        
        # Preparar resultados para serialização (remover objetos modelo)
        serializable_final_results = {}
        for name, result in final_results.items():
            serializable_final_results[name] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
                # Removido 'model' para permitir serialização JSON
            }
        
        # Consolidar resultados
        complete_results = {
            'data_info': {
                'shape': data.shape,
                'features': len(FEATURE_NAMES),
                'varieties': list(VARIETY_NAMES.values())
            },
            'base_models': base_results,
            'optimization': optimization_results,
            'final_evaluation': serializable_final_results,
            'feature_importance': feature_importance,
            'best_model': self._get_best_model(final_results)
        }
        
        return complete_results
    
    def _get_models_config(self):
        """Configuração dos modelos base."""
        from config import RANDOM_SEED
        
        return {
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean',
                weights='uniform'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=RANDOM_SEED
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_SEED
            ),
            'LogisticRegression': LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=RANDOM_SEED
            ),
            'NaiveBayes': GaussianNB(
                var_smoothing=1e-9
            )
        }
    
    def _get_param_grids(self):
        """Grids de parâmetros para otimização."""
        return {
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
    
    def _analyze_base_models(self, X_train, X_test, y_train, y_test):
        """Analisa performance dos modelos base."""
        results = {}
        
        for name, model in self.models_config.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'correct_predictions': int(accuracy * len(y_test)),
                'total_predictions': len(y_test)
            }
            
            print(f"{name}: {accuracy:.4f} ({results[name]['correct_predictions']}/{len(y_test)})")
        
        return results
    
    def _optimize_models(self, X_train, y_train):
        """Otimiza hiperparâmetros dos modelos."""
        optimization_results = {}
        
        for model_name in ['KNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes']:
            print(f"\n🔍 Otimizando {model_name}...")
            
            base_model = self.models_config[model_name]
            param_grid = self.param_grids[model_name]
            
            # Grid Search com validação cruzada
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=CV_FOLDS,
                scoring='accuracy',
                n_jobs=1
            )
            
            grid_search.fit(X_train, y_train)
            
            optimization_results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'total_combinations': len(grid_search.cv_results_['params'])
            }
            
            print(f"  ✅ Melhor CV Score: {grid_search.best_score_:.4f}")
            print(f"  🔧 Parâmetros: {grid_search.best_params_}")
        
        return optimization_results
    
    def _evaluate_optimized_models(self, X_train, X_test, y_train, y_test, optimization_results):
        """Avalia modelos otimizados no conjunto de teste."""
        results = {}
        
        for model_name in self.models_config.keys():
            # Criar modelo otimizado
            if model_name in optimization_results:
                best_params = optimization_results[model_name]['best_params']
                if model_name == 'KNN':
                    model = KNeighborsClassifier(**best_params)
                elif model_name == 'SVM':
                    from config import RANDOM_SEED
                    model = SVC(random_state=RANDOM_SEED, **best_params)
                elif model_name == 'RandomForest':
                    from config import RANDOM_SEED
                    model = RandomForestClassifier(random_state=RANDOM_SEED, **best_params)
                elif model_name == 'LogisticRegression':
                    from config import RANDOM_SEED
                    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, **best_params)
                elif model_name == 'NaiveBayes':
                    model = GaussianNB(**best_params)
            else:
                model = self.models_config[model_name]
            
            # Treinar e avaliar
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Métricas completas
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model  # Para análise posterior
            }
            
            print(f"\n{model_name}:")
            print(f"  Acurácia: {accuracy:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def _analyze_feature_importance(self, final_results):
        """Analisa importância das features (Random Forest)."""
        if 'RandomForest' not in final_results:
            return {}
        
        rf_model = final_results['RandomForest']['model']
        importances = rf_model.feature_importances_
        
        feature_importance = {}
        for i, feature in enumerate(FEATURE_NAMES):
            feature_importance[feature] = importances[i]
        
        # Ordenar por importância
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Ranking de importância:")
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"{i}. {feature}: {importance:.3f}")
        
        return dict(sorted_features)
    
    def _get_best_model(self, final_results):
        """Identifica o melhor modelo baseado na acurácia."""
        best_model = max(final_results.items(), key=lambda x: x[1]['accuracy'])
        return {
            'name': best_model[0],
            'accuracy': best_model[1]['accuracy']
        }


def main():
    """Função principal do script."""
    analyzer = MLModelsAnalyzer()
    results = analyzer.run_complete_analysis()
    
    # Salvar resultados usando o sistema base
    analyzer.save_results(results)
    analyzer.print_success("Análise de modelos ML concluída")
    
    return results


if __name__ == "__main__":
    main()