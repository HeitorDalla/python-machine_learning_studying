# Cenário - Problema para CLASSIFICAÇÃO. Veremos se uma pessoa tem ou não tem uma doença cardíaca
# Objetivo do modelo - Ser capaz de aprender padrões, em seguida, classificar se uma amostra tem uma coisa ou não   

import pandas as pd
import numpy as np

# 0 - Scikit-Learn workflow

# 1 - Deixar os dados prontos para uso
df = pd.read_csv('data/heart-disease.csv')

X = df.drop('target', axis=1)
y = df['target']

# 2 - Escolher o algorítmo para resolver nossos problemas
from sklearn.ensemble import RandomForestClassifier # modelo de classificação

clf = RandomForestClassifier()

# Parâmetros para a utilização do modelo
print(clf.get_params())

# 3 - Ajustar o modelo para os dados de treinamento e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Aqui eu estou treinando o modelo de classificação usando os dados de entrada (X_train)
#  e os rótulos/resultados esperados (y_train), para que ele aprenda a prever a saída a partir dos dados no futuro.
clf.fit(X_train, y_train)

# Fazer uma previsão
y_preds = clf.predict(X_test)
print(y_preds)

# 4 - Avaliar o modelo - avaliar a qualidade de previsões do modelo com os dados de treinamento e 
# os dados de teste
score_trainning = clf.score(X_train, y_train)

# O retorno foi de 100% pois ele foi treinado com todas as colunas e todos os labels, 
# ou seja, ele teve a chance de se corrijir se algo estivesse errado
print(score_trainning)

score_test = clf.score(X_test, y_test)

# O retorno foi de 80% pois ele nunca tinha visto os dados, e não foi treinado com os rótulos
print(score_test)