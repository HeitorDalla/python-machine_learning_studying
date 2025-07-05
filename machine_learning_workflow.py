import pandas as pd
import numpy as np

# Cenário - Problema para CLASSIFICAÇÃO. Veremos se uma pessoa tem ou não tem uma doença cardíaca
# Objetivo do modelo - Ser capaz de aprender padrões, em seguida, classificar se uma amostra tem uma coisa ou não doença   

# Scikit-Learn workflow

# 1 - Deixar os dados prontos para uso

# Separar os dados em caracteristicas e labels (cabeçalho), geralmente em 'X' e 'y'
# Tratar os dados - (limpar os dados -> transformar os dados -> reduzir os dados)
# Converter as colunas categóricas em numéricas

df = pd.read_csv('data/heart-disease.csv')

X = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split # embaralha os dados

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 2 - Escolher o modelo (algoritmo) para resolver nossos problemas
from sklearn.ensemble import RandomForestClassifier # modelo de classificação binária (0 ou 1)

clf = RandomForestClassifier()


# 3 - Ajustar o modelo para os dados de treinamento e teste
clf.fit(X_train, y_train)

# Fazer uma previsão
X_train_prediction = clf.predict(X_train)
print(X_train_prediction)

X_test_prediction = clf.predict(X_test)
print(X_test_prediction)


# 4 - Avaliar o modelo - avaliar a qualidade de previsões do modelo com os dados de treinamento e 
# os dados de teste
score_training = clf.score(X_train, y_train)
print(score_training) # 100%
# O retorno foi de 100% pois ele foi treinado com todas as colunas e todos os labels, 
# ou seja, ele teve a chance de se corrijir se algo estivesse errado

score_test = clf.score(X_test, y_test)
print(score_test) # O retorno foi de 86% pois ele nunca tinha visto os dados, e não foi treinado com os rótulos

# Etapas que pode-se utilizar para avaliar o modelo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, X_test_prediction))

print(confusion_matrix(y_test, X_test_prediction))

print(accuracy_score(y_test, X_test_prediction))


# 5 - Modelo aprimorado - Estimativa para ver se pode-se melhorar o modelo ajustando os hiperparâmetros
np.random.seed(42)

for i in range (10, 100, 10):
    print("Testando os modelos com: {} estimadores..." .format(i))

    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)

    print("Testando a precisão do modelo com os dados de 'test': {:.2f}%" .format(clf.score(X_test, y_test) * 100))

    print('')


# 6 - Exportar e importar um modelo
import pickle

pickle.dump(clf, open('machine_learning/random_first_model.pkl', 'wb'))

loaded_model = pickle.load(open('machine_learning/random_first_model.pkl', 'rb'))

# Bônus = Ignorar avisos
import warnings

warnings.filterwarnings('ignore') # vai ignorar todos os avisos que ocorrerem no código