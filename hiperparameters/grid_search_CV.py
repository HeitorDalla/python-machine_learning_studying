import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Parâmetros de Utilização

## max_depth
## max_features
## min_samples_leaf
## min_samples_split
## n_estimators

# Função para retornar a avaliação do modelo por meio de Métricas
def evaluated_metrics (y_true, y_preds):
    accuracy = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)

    metrics = {
        'accuracy': round(accuracy, 2),
        'f1': round(f1, 2),
        'precision': round(precision, 2),
        'recall' : round(recall, 2)
    }

    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metrics


# Tratamento dos Dados
carregamento = load_breast_cancer()

dataframe = pd.DataFrame(carregamento['data'], columns=carregamento['feature_names'])

dataframe['target'] = carregamento['target']

X = dataframe.drop('target', axis=1)
y = dataframe['target']

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42
                                                    )

model = RandomForestClassifier(n_jobs=1, random_state=42)

from sklearn.model_selection import GridSearchCV

# Parâmetros que vão ser utilizados
grid = {
    'n_estimators': [100, 200, 500], 
    'min_samples_split': [6], 
    'min_samples_leaf': [1, 2], 
    'max_features': ['sqrt'],
    'max_depth': [20]
}

# Modelo parametrizado
gs_model = GridSearchCV(estimator=model,
                        param_grid=grid,
                        cv=5,
                        verbose=2)

# Treinamento do modelo
modelo_treinado = gs_model.fit(X_train, y_train)

# Mostrar a melhor performance
print(gs_model.best_params_)

# Previsão dos dados
gs_y_preds = gs_model.predict(X_test)

# Avaliação de performance com métricas
gs_metrics = evaluated_metrics(y_test, gs_y_preds)
print(gs_metrics)


# Exportar o modelo finalizado
from joblib import dump, load

dump(gs_model, filename='models_final/grid_search_model.joblib')


# Importar um modelo finalizado
loaded_joblib_model = load(filename="models_final/grid_search_model.joblib")