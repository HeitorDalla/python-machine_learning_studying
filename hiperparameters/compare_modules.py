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

from sklearn.model_selection import RandomizedSearchCV

# Instanciar o Modelo à ser usado
model = RandomForestClassifier(n_jobs=1, random_state=42) # quantas CPUs o algorítmo vai utilizar para realizar as tarefas

grid = {
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ['sqrt', 'log2'],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4, 6],
    "n_estimators": [10, 100, 200, 500]
}

rs_model = RandomizedSearchCV(model, # modelo que vai utilizar
                              param_distributions=grid, # os parâmetros que vai utilizar
                              n_iter=10, # número de modelos para exmperimentar
                              cv=5, # cruzada - faz divisões diferentes em cada vez
                              verbose=2)

# Treinamento dos dados parametrizados
modelo_treinado = rs_model.fit(X_train, y_train)

# Ver o melhor resultado da parametrização
print(rs_model.best_params_)

# Fazer previsões com o modelo parametrizado
rs_y_preds = rs_model.predict(X_test)

# Classificar o modelo com as métricas
rs_metrics = evaluated_metrics(y_test, rs_y_preds)
print(rs_metrics)


model2 = RandomForestClassifier(n_jobs=1, random_state=42)

from sklearn.model_selection import GridSearchCV

# Parâmetros que vão ser utilizados
grid2 = {
    'n_estimators': [100, 200, 500], 
    'min_samples_split': [6], 
    'min_samples_leaf': [1, 2], 
    'max_features': ['sqrt'],
    'max_depth': [20]
}

# Modelo parametrizado
gs_model = GridSearchCV(estimator=model,
                        param_grid=grid2,
                        cv=3,
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


# Comparação entre os Modelos
compare_metrics = pd.DataFrame({
    'random search': rs_metrics,
    'grid search': gs_metrics
})
compare_metrics.plot.bar(figsize=(10, 8))
plt.show()