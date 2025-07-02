import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Modelos de treinamento
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

carregamento = load_iris()

dataframe = pd.DataFrame(carregamento['data'], columns=carregamento['feature_names'])

dataframe['target'] = carregamento['target']

print(dataframe)

# Separar os dados em variáveis dependentes e independentes
X = dataframe.drop('target', axis=1)
y = dataframe['target']

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

# Instanciando o modelo de classificacao
model = RandomForestClassifier()

 # Treinamento dos dados
training = model.fit(X_train, y_train)

# Predict dos dados
y_preds = model.predict(X_test)

# Avaliação do modelo
from sklearn.metrics import accuracy_score

# Checar a acurácia do modelo, ou seja, porcentagem de acertos em cima dos exemplos
avaliacao = accuracy_score(y_test, y_preds)
print(avaliacao)