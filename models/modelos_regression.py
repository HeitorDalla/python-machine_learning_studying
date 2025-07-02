import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Modelos de regressão
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

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

# Instanciar o modelo e ajusta-lo
model = RandomForestRegressor()

# Treinar o modelo
training = model.fit(X_train, y_train)

# Fazer uma previsão
y_preds_train = model.predict(X_train)

y_preds_test = model.predict(X_test)

# Check score of the model (on the test set)
from sklearn.metrics import r2_score

# Indica quanto da variação dos valores reais o modelo de regressão conseguiu explicar
test_score = r2_score(y_test, y_preds_test)
print(test_score)