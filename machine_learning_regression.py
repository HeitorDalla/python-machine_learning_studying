import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Modelos
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

# Tratamento dos Dados
# Análise dos Dados
housing = fetch_california_housing()

dataframe = pd.DataFrame(housing['data'], columns=housing['feature_names'])

dataframe['target'] = housing['target']

# Separar os dados para treinamento e teste
X = dataframe.drop('target', axis=1)
y = dataframe['target']

# Importar algoritmo de modelo para o aprendizado de máquina
np.random.seed(42)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_teste = train_test_split(X,
                                                     y,
                                                     test_size=0.2)

# Instanciar o modelo e ajusta-lo
model = Ridge()

trainning = model.fit(X_train, y_train)

# Check score of the model (on the test set)
testing = model.score(X_test, y_teste) # coeficiente de correlação entre variáveis

print(testing)