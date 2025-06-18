"""Módulo para treinamento e avaliação de modelos."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from config import RANDOM_SEED


def get_models():
    """
    Retorna dicionário com modelos para treinamento.
    
    Returns:
        dict: Dicionário com modelos
    """
    models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    }
    
    return models


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Treina e avalia um modelo.
    
    Parameters:
        model: Modelo scikit-learn
        X_train: Características de treino
        y_train: Rótulos de treino
        X_test: Características de teste
        y_test: Rótulos de teste
        model_name (str): Nome do modelo
        
    Returns:
        dict: Resultados da avaliação
    """
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✅ {model_name}: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Treina e avalia todos os modelos.
    
    Parameters:
        X_train: Características de treino
        y_train: Rótulos de treino
        X_test: Características de teste
        y_test: Rótulos de teste
        
    Returns:
        dict: Resultados de todos os modelos
    """
    models = get_models()
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, model_name
        )
    
    return results


def get_hyperparameter_grids():
    """
    Retorna grids otimizados para balance velocidade/qualidade.
    
    Returns:
        dict: Grids de hiperparâmetros otimizados
    """
    param_grids = {
        'KNN': {
            'n_neighbors': [3, 5, 7],  # Reduzido de 5 para 3 valores
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']  # Removido minkowski
        },
        'SVM': {
            'C': [1, 10, 100],  # Reduzido, removido 0.1
            'gamma': ['scale', 0.01, 0.1],  # Foco nos mais promissores
            'kernel': ['rbf', 'linear']  # Removido poly (raramente melhor)
        },
        'Random Forest': {
            'n_estimators': [50, 100],  # Reduzido, 200 raramente melhora muito
            'max_depth': [None, 20],  # Simplificado
            'min_samples_split': [2, 5]  # Reduzido
        }
    }
    
    return param_grids


def optimize_model(model_name, X_train, y_train, X_test, y_test):
    """
    Otimiza hiperparâmetros de um modelo usando Grid Search.
    
    Parameters:
        model_name (str): Nome do modelo
        X_train: Características de treino
        y_train: Rótulos de treino
        X_test: Características de teste
        y_test: Rótulos de teste
        
    Returns:
        dict: Resultados da otimização
    """
    param_grids = get_hyperparameter_grids()
    
    if model_name not in param_grids:
        print(f"Otimização não disponível para {model_name}")
        return None
    
    print(f"\nOtimizando {model_name}...")
    
    # Criar novo modelo
    if model_name == 'KNN':
        base_model = KNeighborsClassifier()
    elif model_name == 'SVM':
        base_model = SVC(random_state=RANDOM_SEED)
    elif model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=RANDOM_SEED)
    
    # Grid Search
    grid_search = GridSearchCV(
        base_model, 
        param_grids[model_name], 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Melhor modelo
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Avaliar modelo otimizado
    y_pred_opt = best_model.predict(X_test)
    accuracy_opt = accuracy_score(y_test, y_pred_opt)
    precision_opt = precision_score(y_test, y_pred_opt, average='weighted')
    recall_opt = recall_score(y_test, y_pred_opt, average='weighted')
    f1_opt = f1_score(y_test, y_pred_opt, average='weighted')
    
    print(f"\nMelhores parâmetros para {model_name}:")
    print(best_params)
    print(f"\nResultados otimizados:")
    print(f"Acurácia: {accuracy_opt:.4f}")
    
    return {
        'model': best_model,
        'best_params': best_params,
        'accuracy': accuracy_opt,
        'precision': precision_opt,
        'recall': recall_opt,
        'f1_score': f1_opt,
        'predictions': y_pred_opt,
        'confusion_matrix': confusion_matrix(y_test, y_pred_opt)
    }


def perform_cross_validation(model, X_train, y_train, cv=5):
    """
    Realiza validação cruzada.
    
    Parameters:
        model: Modelo treinado
        X_train: Características de treino
        y_train: Rótulos de treino
        cv (int): Número de folds
        
    Returns:
        dict: Resultados da validação cruzada
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    
    return {
        'scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }