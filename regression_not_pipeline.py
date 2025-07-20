# Bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Estruturação dos Dados
from sklearn.model_selection import train_test_split # separação para treinamento e teste
from sklearn.impute import SimpleImputer # substitui valores ausentes com uma estratégia definida (média, moda, valor que mais aparece)
from sklearn.compose import ColumnTransformer # permite aplicar transformacões diferentes para diferentes colunas
from sklearn.preprocessing import OneHotEncoder # codificador para transformar categorias em números

# Modelos de Algorítmos de Regressão
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

# Métricas de Avaliação
from sklearn.model_selection import cross_val_score # validação cruzada, medindo a performance em várias divisões
from sklearn.metrics import r2_score # o quão bem o modelo explica os dados
from sklearn.metrics import mean_absolute_error # calcula o erro médio absoluto
from sklearn.metrics import mean_squared_error # calcula o erro quadrático médio, penalizando erros graves

# Ajuste de Hiperparâmetros - métodos para encontrar a melhor combinação de parâmetros, 
# testando várias combinações automaticamente com a validação cruzada (CV)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV # testa todas as combinações possíveis dos parâmetros fornecidos

# Salvar o modelo
import pickle

# Função para retornar a avaliação do modelo por meio de Métricas
def evaluated_metrics (y_true, y_preds):
    r2 = r2_score(y_true, y_preds)
    mean_absolute = mean_absolute_error(y_true, y_preds)
    mean_squared = mean_squared_error(y_true, y_preds)

    metrics = {
        'r2': round(r2, 2),
        'mean_absolute_error': round(mean_absolute, 2),
        'mean_squared_error': round(mean_squared, 2)
    }

    print(f"R2: {r2 * 100:.2f}%")
    print(f"Mean_absolute_error: {mean_absolute:.2f}")
    print(f"Mean_squared_error: {mean_squared:.2f}")

    return metrics


# 1 - Exploração inicial dos dados

# Objetivo - Antes de qualquer modelo, é preciso conhecer os dados para saber se eles estão utilizáveis
#  e se possuem informações suficientes para prever sua variável-alvo.

# Carregar dataset
df = pd.read_csv("data/car-sales-extended-missing-data.csv")
print(df)

# Análise do dataset
print(df.info()) # verificar os tipos de dados presentes
print(df.describe()) # estatísticas básicas do dataset
print(df.head()) # ver as linhas iniciais para possíveis modificações de normalizações

# Visualizações com histogramas e boxplots para entender a relação entre os dados

# Verificar a correlação dos dados
correlacao = df.corr(numeric_only=True)
print(correlacao)


# 2 - Pré-processamento dos dados e Engenharia dos atributos

# Objetivo - Os modelos não conseguem trabalhar com dados sujos, portanto, 
# deve-se tratar para colocar o treinamento em prática

# Tratar os valores nulos
print(df.isna().sum()) # verifica quantos valores ausentes há em cada coluna

# Remover apenas linhas que possuvem valores ausentes na categoria alvo
df = df.dropna(subset=["Price"])
print(df)

# Separar os dados em colunas dependentes e independentes
X = df.drop("Price", axis=1) # colunas de alimento para o modelo de AI
y = df["Price"] # coluna para o modelo prever

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% para testes
                                                    random_state=42) # a divisão vai ser a mesma sempre que rodar

# Transformar os valores nulos

# Definir as colunas por tipos para aplicar pré-processamentos diferentes
colunas_categoricas = ["Make", "Colour"] # texto - número
coluna_door = ["Doors"] # média
colunas_numericas = ["Odometer (KM)"] # tratada como valor fixo

# Criar os imputadores
valor_substituto_categorica = SimpleImputer(strategy="constant", fill_value="missing") # usado para substituir 'missing' em colunas categóricas
valor_substituto_door = SimpleImputer(strategy="constant", fill_value=4) # usado para substituir para a 4 (é o valor mais comum) na coluna 'Doors'
valor_substituto_numericos = SimpleImputer(strategy="mean") # usado para colunas contínuas (numéricas)

### Criar o imputer (aplica diferentes técnicas em diferentes colunas)
# O ColumnTransformer permite processar diferentes grupos de colunas com técnicas diferentes
imputer = ColumnTransformer([ # pré-processador único que aplica para cada grupo de colunas com técnicas diferentes
    ("valor_substituto_categorica", valor_substituto_categorica, colunas_categoricas),
    ("valor_substituto_door", valor_substituto_door, coluna_door),
    ("valor_substituto_numericos", valor_substituto_numericos, colunas_numericas)
])

# Preencher os valores ausentes (coluna por coluna) usando a estratégia definida ('missing', 4, mean)
X_train_transformed = imputer.fit_transform(X_train) # treina o modelo com as técnias ('missing', 4, mean) e aplica os valores aprendidos nos valores ausentes
X_test_transformed = imputer.transform(X_test) # usa os mesmos valores aprendidos no treino e apenas preenche as técnias ('missing', 4, mean)

# Transformar as colunas categoricas em numericas
variaveis_categoricas = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder() # objeto para criar um codificador de texto para números binários, criando uma coluna com 0 ou 1

transformer = ColumnTransformer([('one_hot', # nome da qualquer para transformação
                                 one_hot, # objeto criado para codificação
                                 variaveis_categoricas)], # lista para as colunas que serão codificadas
                                 remainder='passthrough') # define o que fazer com as colunas que não estão nas lista de transformação, no caso, mantém no conjunto de dados

# Como os dados foram transformados em arrays pelo imputer, precisamos recriar os DataFrames
# com os mesmos nomes de colunas para aplicar o OneHotEncoder baseado nos nomes
X_train_df = pd.DataFrame(X_train_transformed, columns=colunas_categoricas + coluna_door + colunas_numericas)
X_test_df = pd.DataFrame(X_test_transformed, columns=colunas_categoricas + coluna_door + colunas_numericas)

# Aplicar a codificação
X_train_final = transformer.fit_transform(X_train_df) # aplica a codificação para o treino
X_test_final = transformer.transform(X_test_df) # aplica a codificação para os testes


# 3 - Escolha de um modelo inicial e utilização de hiperparâmetros padrões

# Objetivo - Ter o melhor modelo para resolver o problema

# Instanciando o modelo para utilização
model = RandomForestRegressor(n_jobs=1, random_state=42)

# Treinamento do modelo
training = model.fit(X_train_final, y_train)

# Fazendo previsões do modelo
y_preds = model.predict(X_test_final)
print(y_preds)


# 4 - Avaliação dom Métricas

# Objetivo - medir o desempenho incial do modelo

# r2_score - indica quanto da variação dos valores reais o modelo de regressão conseguiu explicar
default_cross_score = cross_val_score(model, X_train_final, y_train, cv=2, scoring=None) # array de 10 treinamentos diferentes
print(default_cross_score)

r2_cross_score = cross_val_score(model, X_train_final, y_train, cv=2, scoring='r2') # array de 10 treinamentos diferentes
print(r2_cross_score)

test_score = r2_score(y_test, y_preds) # valor de um treinamento
print(test_score)

# São equivalentes ao mesmo resultado.


# MAE (Mean absolute error) - pega o erro médio entre o valor real e a previsão, sem direção (- ou +)
mean = mean_absolute_error(y_test, y_preds)
print(mean)
# Interpretação - O modelo erra (média) a cada previsão.

# Fazendo manualmente
dataframe_difference = pd.DataFrame(data={'reais_valores': y_test,
                                          'previsao_valores':y_preds})

dataframe_difference['differences'] = dataframe_difference['previsao_valores'] - dataframe_difference['reais_valores']
print(dataframe_difference)

print(np.abs(dataframe_difference['differences']).mean()) # abs - pois o valor deve ser absoluto, ou seja, ignorar ser positivo ou negativo


# MSE (Mean Squared Error) - o erro médio, porém ao quadrado, penalizando ainda mais os erros grandes do modelo
from sklearn.metrics import mean_squared_error

squared = mean_squared_error(y_test, y_preds)
print(squared)


# 5 - Ajuste de Hiperparâmetros

# Objetivo - Melhorar o desempenho do modelo escolhido ajustando as configurações internas

# OBS - Isso acontece APÓS o primeiro treinamento e avaliação inicial, pois, 
# testar hiperparâmetros antes de testa-lo poda gastar tempo atoa


# RandomizedSearchCV

# Escolher os hiperparâmetros que vão ser ajustados
grid_rs = {
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ['sqrt', 'log2'],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4, 6],
    "n_estimators": [10, 100, 200, 500]
}

# Instanciando o novo modelo com os hiperparâmetros ajustados
rs_model = RandomizedSearchCV(model, # modelo que vai utilizar
                              param_distributions=grid_rs, # os parâmetros que vai utilizar
                              n_iter=10, # número combinações aleatórias à testar
                              cv=5, # divisão cruzada - faz divisões diferentes em cada vez
                              verbose=2) # nível de detalhamento dos logs no console

# Treinamento do novo modelo com os dados parametrizados
modelo_treinado = rs_model.fit(X_train_final, y_train)

# Ver o melhor resultado da parametrização
print(rs_model.best_params_)

# Fazer previsões com o modelo parametrizado
rs_y_preds = rs_model.predict(X_test_final)

# Classificar o modelo com as métricas
rs_metrics = evaluated_metrics(y_test, rs_y_preds)
print(rs_metrics)


# GridSearchCV

# Escolhre os parâmetros que vão ser utilizados
grid_gs = {
    'n_estimators': [100, 200, 500],
    'min_samples_split': [6],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'max_depth': [20]
}

# Modelo parametrizado
gs_model = GridSearchCV(estimator=model, # modelo que vai utilizar
                        param_grid=grid_gs, # os parâmetros que vai utilizar
                        cv=5, # divisão cruzada - faz divisões diferentes em cada vez
                        verbose=2)

# Treinamento do modelo parametrizado
modelo_treinado = gs_model.fit(X_train_final, y_train)

# Mostrar a melhor performance
print(gs_model.best_params_)

# Previsão dos dados
gs_y_preds = gs_model.predict(X_test_final)

# Avaliação de performance com métricas
gs_metrics = evaluated_metrics(y_test, gs_y_preds)
print(gs_metrics)


# 6 - Salvar o Modelo

# Objetivo - salvar para testar em outros dispositivos

# Salvando o modelo finalizado
pickle.dump(rs_model, open('models_final/randomized_search_model_not_pipeline.pkl', 'wb'))


# 7 - Interpretação e Comunicação dos Resultados

# Objetivo - Explicar o desempenho do modelo para stakeholders

# Mostrar métricas

# Plotar o real e o previsto

# Ver ondo o modelo erra e explicar o motivo