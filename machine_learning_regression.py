import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

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

