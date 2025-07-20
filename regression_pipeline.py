# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Estruturação dos Dados
from sklearn.model_selection import train_test_split # separação para treinamento e teste
from sklearn.pipeline import Pipeline # permite encadear transformações e modelos em uma sequência
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

np.random.seed(42) # para garantir a reprodutibilidade dos resultados

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


# Pipeline - Usado para encadear transformações e modelos em uma sequência,
# facilitando o pré-processamento e treinamento do modelo.

# 1 - Tratar dos valores ausentes
# 2 - Codificar as colunas categóricas em colunas numéricas
# 3 - Treinar um modelo

# Para cada grupo de colunas, deve-se criar um Pipeline específico para cada transformação, ou seja,
# um pipeline para valores ausentes e outro para aplicar codificação
categorical_features = ['Make', 'Colour'] # colunas para transformar em numéricas
categorical_transformer = Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='constant', fill_value='missing')), # substitui valores ausentes por 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # codifica as colunas
])

door_feature = ['Doors']
door_transformer = Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='constant', fill_value=4)) # substitui valores ausentes pelo valor que mais aparece (4)
])

numeric_feature = ['Odometer (KM)'] # colunas numéricas
numeric_transformer = Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='mean')) # substitui valores ausentes pela média da coluna
])

# Fazendo o pré-processamento dos dados (valores ausentes e codificação das colunas)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features), # em colunas categóricas
        ('door', door_transformer, door_feature), # em colunas de valores ausentes
        ('num', numeric_transformer, numeric_feature) # em colunas numéricas
    ]
)

# Criando um Pipeline final que une o pré-processamento e modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor), # pré-processamento dos dados
    ('model', RandomForestRegressor(random_state=42)) # modelo
])

# Sepando os dados em treinamento e teste
X = df.drop('Price', axis=1) # excluindo a coluna alvo para os dados de treinamento
y = df['Price'] # definindo a coluna alvo para os testes

# Separando os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Treinando o modelo com os dados
training = model.fit(X_train, y_train)

# Fazendo previsões com o modelo de teste
y_preds = model.predict(X_test)

# Avaliando o modelo
y_score = model.score(X_test, y_test)

print("A acurácia do modelo é: {}" .format(y_score))


# 4 - Avaliação dom Métricas

# Objetivo - medir o desempenho incial do modelo

# r2_score - indica quanto da variação dos valores reais o modelo de regressão conseguiu explicar
default_cross_score = cross_val_score(model, X_train, y_train, cv=2, scoring=None) # array de 10 treinamentos diferentes
print(default_cross_score)

r2_cross_score = cross_val_score(model, X_train, y_train, cv=2, scoring='r2') # array de 10 treinamentos diferentes
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
    'preprocessor__num__inputer__strategy': ['mean', 'median'], # estratégia de imputação
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__max_features": ['sqrt', 'log2'],
    "model__min_samples_leaf": [1, 2, 4],
    "model__min_samples_split": [2, 4, 6],
    "model__n_estimators": [10, 100, 200, 500]
}

# Instanciando o novo modelo com os hiperparâmetros ajustados
rs_model = RandomizedSearchCV(model, # modelo que vai utilizar
                              param_distributions=grid_rs, # os parâmetros que vai utilizar
                              n_iter=10, # número combinações aleatórias à testar
                              cv=5, # divisão cruzada - faz divisões diferentes em cada vez
                              verbose=2) # nível de detalhamento dos logs no console

# Treinamento do novo modelo com os dados parametrizados
modelo_treinado_rs = rs_model.fit(X_train, y_train)

# Ver o melhor resultado da parametrização
print(rs_model.best_params_)

# Fazer previsões com o modelo parametrizado
rs_y_preds = rs_model.predict(X_test)

# Classificar o modelo com as métricas
rs_metrics = evaluated_metrics(y_test, rs_y_preds)
print(rs_metrics)


# GridSearchCV

# Escolhre os parâmetros que vão ser utilizados
grid_gs = {
    'preprocessor__num__inputer__strategy': ['mean', 'median'], # estratégia de imputação
    'model__n_estimators': [100, 1000],
    'model__min_samples_split': [6],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt'],
    'model__max_depth': [20]
}


# Modelo parametrizado
gs_model = GridSearchCV(estimator=model, # modelo que vai utilizar
                        param_grid=grid_gs, # os parâmetros que vai utilizar
                        cv=5, # divisão cruzada - faz divisões diferentes em cada vez
                        verbose=2)

# Treinamento do modelo parametrizado
modelo_treinado_gs = gs_model.fit(X_train, y_train)

# Mostrar a melhor performance
print(gs_model.best_params_)

# Previsão dos dados
gs_y_preds = gs_model.predict(X_test)

# Avaliação de performance com métricas
gs_metrics = evaluated_metrics(y_test, gs_y_preds)
print(gs_metrics)


# 6 - Salvar o Modelo

# Objetivo - salvar para testar em outros dispositivos

# Salvando o modelo finalizado
pickle.dump(rs_model, open('models_final/randomized_search_model_pipeline.pkl', 'wb'))


# 7 - Interpretação e Comunicação dos Resultados

# Objetivo - Explicar o desempenho do modelo para stakeholders

# Mostrar métricas

# Plotar o real e o previsto

# Ver ondo o modelo erra e explicar o motivo