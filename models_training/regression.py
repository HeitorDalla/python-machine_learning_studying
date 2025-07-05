import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Modelos de regressão
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

# Tratamento dos Dados
# Análise dos Dados
housing = fetch_california_housing()

dataframe = pd.DataFrame(housing['data'], columns=housing['feature_names'])

dataframe['target'] = housing['target']

# Separar os dados em variáveis dependentes e independentes
X = dataframe.drop('target', axis=1)
y = dataframe['target']

# Importar algoritmo de modelo para o aprendizado de máquina
np.random.seed(42)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size=0.2)


# Instanciar o modelo e ajusta-lo para o treinamento
model = RandomForestRegressor()

# Treinar o modelo
training = model.fit(X_train, y_train)

# Fazer uma previsão
y_preds_test = model.predict(X_test)


# Avaliação do Modelo por meio de Métricas

# r2_score - 
from sklearn.model_selection import cross_val_score # Vai gerar 'cv' treinamentos diferentes para o mesmo conjunto de dados
from sklearn.metrics import r2_score # Indica quanto da variação dos valores reais o modelo de regressão conseguiu explicar

default_cross_score = cross_val_score(model, X, y, cv=2, scoring=None) # array de 10 treinamentos diferentes

r2_cross_score = cross_val_score(model, X, y, cv=2, scoring='r2') # array de 10 treinamentos diferentes

test_score = r2_score(y_test, y_preds_test) # valor de um treinamento

# São equivalentes ao mesmo resultado.


# MAE (Mean absolute error) - pega o erro médio entre o valor real e a previsão, sem direção (- ou +)
from sklearn.metrics import mean_absolute_error

# Utilizando a métrica
mean = mean_absolute_error(y_test, y_preds_test)
print(mean)
# Interpretação - O modelo erra {} a cada previsão.

# Fazendo manualmente
dataframe_difference = pd.DataFrame(data={'reais_valores': y_test,
                                          'previsao_valores':y_preds_test})

dataframe_difference['differences'] = dataframe_difference['previsao_valores'] - dataframe_difference['reais_valores']
print(dataframe_difference)

print(np.abs(dataframe_difference['differences']).mean()) # abs - pois o valor deve ser absoluto, ou seja, ignorar ser positivo ou negativo


# MSE (Mean Squared Error) - o erro médio, porém ao quadrado, penalizando ainda mais os erros grandes do modelo
from sklearn.metrics import mean_squared_error

squared = mean_squared_error(y_test, y_preds_test)
print(squared)