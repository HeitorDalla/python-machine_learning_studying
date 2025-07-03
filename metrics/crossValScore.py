import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Modelos de treinamento
from sklearn.ensemble import RandomForestClassifier

# np.random.seed(42)

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
from sklearn.model_selection import cross_val_score

# Checar a acurácia do modelo, ou seja, porcentagem de acertos em cima dos exemplos
default = cross_val_score(model, X, y, cv=10, scoring=None) # cv - quantidade que eu quero treinar

# Média do modelo 'cross_val_score'
print(np.mean(default))
print('-----------------')

# Comparação da classificação cruzada X score normal
print(model.score(X_test, y_test))
print(np.mean(default))