import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# Modelos de treinamento
from sklearn.ensemble import RandomForestClassifier


# Tratamento dos Dados
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


# Instanciando o modelo de classificacao, treinando os dados e fazendo previsões
model = RandomForestClassifier()

# Treinamento dos dados
training = model.fit(X_train, y_train)

# Previsão dos dados
y_preds = model.predict(X_test)
print(y_preds[:10]) # para cada linha, vai mostrar se é 0 ou 1

print('-----------------------')

y_proba = model.predict_proba(X_test)
print(y_proba[:10]) # mostra as 10 primeiras linhas com as possibilidades de ser 0 ou 1
print(y_proba[:10, 1]) # o 1 significa que que vai mostrar as 10 possibilidades de ser 1

print('-----------------------')

### Diferença de 'predict' X 'predict_proba'
# A principal diferença do 'predict' para o 'predict_proba' é que o predict vai mostrar 0 ou 1
# , ou seja, para cada linha mostrar se vai ser 0 ou 1.
# Já o 'predict_proba' vai mostrar, para cada linha, quais as probabilidades de ser 0 ou 1, ou seja, 
# não escolhe um direto.


# Avaliação do Modelo por meio de Métricas

# accuracy_score - Proporção de Acertos sobre o total de dados
from sklearn.model_selection import cross_val_score # modelo para diversas métricas
from sklearn.metrics import accuracy_score # Checar a acurácia do modelo, ou seja, porcentagem de acertos em cima dos exemplos

default_cross_score = cross_val_score(model, X, y, cv=10, scoring=None) # array de 10 treinamentos diferentes (vai retornar a acurácia padrão (score))

accuracy_cross_score = cross_val_score(model, X, y, cv=10, scoring='accuracy') # array de 10 treinamentos diferentes (vai retornar a acurácia padrão (score))

test_score = accuracy_score(y_test, y_preds) # vai retornar a acurácia padrão (score)

### São equivalentes ao mesmo resultado.


# Avaliação da métrica 'roc_curve'
from sklearn.metrics import roc_curve # distingue as duas classes (0 ou 1)

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
plt.plot(fpr, tpr, color='red', label='ROC')

# Personalizando a plotagem
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.show()


# Avaliação da métrica 'roc_auc_score'
from sklearn.metrics import roc_auc_score

# Calculo
auc_roc_score = roc_auc_score(y_test, y_proba[:, 1])
print(auc_roc_score)


# confusion_matrix - Mostra os acertos e erros (análise detalhada)
from sklearn.metrics import confusion_matrix

metric_confusion = confusion_matrix(y_test, y_preds)
print(metric_confusion)

# Tabela da Métrica
table_confusion = pd.crosstab(y_test,
                              y_preds,
                              rownames=['Rótulo Real'],
                              colnames=['Rótulo Previsto'])
print(table_confusion)

# Gráfico da Métrica
from sklearn.metrics import ConfusionMatrixDisplay

# Usando o from_estimator - Usa todos os dados, sem separação de treinamento e teste
grafico = ConfusionMatrixDisplay.from_estimator(estimator=model,
                                                X=X,
                                                y=y)
plt.show()

# Usando o from_predictions - Usa os dados já filtrados em treinamento e teste
grafico2 = ConfusionMatrixDisplay.from_predictions(y_true=y_test,
                                                   y_pred=y_preds)
plt.show()


# classification_report - Usa diferentes métricas para avaliar um modelo
from sklearn.metrics import classification_report

report_predictions = classification_report(y_test, y_preds)
print(report_predictions)

# precision - verdadeiro positivo / verdadeiro positivo + falso positivo: (acertos em quando o modelo prevê 1)
# recall - verdadeiro positivo / verdedeiro positivo + falso negativo: (acertos quando o rótulo verdadeiro é 1)
# f1-score - representa a média harmônica entre o 'precision' e 'recall'