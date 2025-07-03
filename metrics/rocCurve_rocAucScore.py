import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer # 2 classes
from sklearn.model_selection import train_test_split
# Modelos de treinamento
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

carregamento = load_breast_cancer()

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
y_proba = model.predict_proba(X_test)
print(y_proba[:10]) # mostra as 10 primeiras linhas com as possibilidades de ser 0 ou 1
print(y_proba[:10, 1]) # o 1 significa que que vai mostrar as 10 possibilidades de ser 1

print('-----------------------')

y_preds = model.predict(X_test)
print(y_preds[:10]) # significa que vai apenas mostrar qual é a probabilidade maior de ser 1 ou 0

print('-----------------------')
print('-----------------------')

### Diferença de 'predict' X 'predict_proba'
# A principal diferença do 'predict' para o 'predict_proba' é que o predict vai mostrar 0 ou 1
# , ou seja, para cada linha mostrar se vai ser 0 ou 1.
# Já o 'predict_proba' vai mostrar, para cada linha, quais as probabilidades de ser 0 ou 1, ou seja, 
# não escolhe um direto.

# Avaliação do modelo 'roc_curve'
from sklearn.metrics import roc_curve

def grafico_roc_curve (fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')

    # Personalizando a plotagem
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend()
    plt.show()

# Calculo
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
# fpr - alarme falso
# tpr - acerto nos positivos
# thresholds - limiar de probabilidade usado para calcular 0 ou 1

print(fpr)
print('-----------------------')
print(tpr)
print('-----------------------')
print(thresholds)

# Gráfico da curva
grafico_roc_curve(fpr, tpr)

# Avaliação do modelo 'roc_auc_score'
from sklearn.metrics import roc_auc_score

# Calculo
auc_roc_score = roc_auc_score(y_test, y_proba[:, 1])
print(auc_roc_score)