import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Deixar os dados prontos para uso

# Separar os dados em caracteristicas e labels (cabeçalho), geralmente em 'X' e 'y'
# Tratar os dados - (limpar os dados -> transformar os dados -> reduzir os dados)
# Converter as colunas categóricas em numéricas

df = pd.read_csv("data/car-sales-extended-missing-data.csv")

# Tratamento de dados
print(df.isna().sum()) # verifica quantos valores ausentes há em cada coluna

# Preencher com valores lógicos para interferir o menos possível
df['Make'] = df['Make'].fillna('missing') # preenche todos os valores nulos com 'missing'
df['Colour'] = df['Colour'].fillna('missing') # preenche todos os valores nulos com 'missing'
df['Odometer (KM)'] = df['Odometer (KM)'].fillna(df['Odometer (KM)'].mean()) # preenche todos os valores nulos com a média da coluna
df['Doors'] = df['Doors'].fillna(4) # preenche todos os valores nulos com o maior que aparece (4)
df = df.dropna() # o melhor a se fazer é excluir os dados faltantes, pois não temos base de previsão para a coluna alvo

# Separo as variáveis independentes da variável alvo
X = df.drop('Price', axis=1)
y = df['Price']

# Separar o dataframe em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, # colocando o dataframe numérico apenas
                                                    y, 
                                                    test_size=0.2)

# Verificar dataframe - Existem dois campos que sao objetos, tem que transformar em numéricos
print(df.info())

# Transformar as colunas categoricas em numericas
from sklearn.preprocessing import OneHotEncoder # transformar categorias em números
from sklearn.compose import ColumnTransformer # aplica essa transformação nas colunas escolhidas

variaveis_categoricas = ['Make', 'Colour', 'Doors'] # Doors entra por ser uma variável categorizadora numérica

one_hot = OneHotEncoder() # cria um codificador de texto para números binários, criando uma coluna com 0 ou 1
transformer = ColumnTransformer([('one_hot', # nome da transformação
                                 one_hot, 
                                 variaveis_categoricas)], # lista como o nome das variáveis categóricas
                                 remainder='passthrough') # define o que fazer com as colunas que não estão nas lista de transformação, no caso, mantém no conjunto de dados

X_train_transformed = transformer.fit_transform(X_train) # analisa os dados e faz a codificação
X_test_transformed = transformer.transform(X_test)