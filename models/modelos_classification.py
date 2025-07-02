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

# Implementar modelo para usabilidade
model = SVC()

training = model.fit(X_train, y_train)

# Checar a classificação do modelo com o conjunto de testes
testing = model.score(X_test, y_test)
print(testing)


# Instanciando o modelo de classificacao
model2 = RandomForestClassifier()

 # Treinamento dos dados
training2 = model2.fit(X_train, y_train)

# Predict dos dados
y_preds = model2.predict(X_test)

# Avaliação do modelo
from sklearn.metrics import accuracy_score
avaliacao = accuracy_score(y_test, y_preds)
print(avaliacao)

# Checar a classificacao do modelo com o conjunto de testes
testing2 = model2.score(X_test, y_test)
print(testing2)