import pandas as pd
import numpy as np

# Scikit-Learn workflow

# 1 - Deixar os dados prontos para uso

# Separar os dados em caracteristicas e labels (cabeçalho), geralmente em 'X' e 'y'
# Tratar os dados - (limpar os dados -> transformar os dados -> reduzir os dados)
# Converter as colunas categóricas em numéricas

# Carregar dataset
df = pd.read_csv("data/car-sales-extended-missing-data.csv")

# Verifica quantos valores ausentes há em cada coluna
print(df.isna().sum())

# Remover apenas linhas ausentes da categoria alvo (Price)
df = df.dropna(subset=["Price"])

# Separar os dados em colunas dependentes e independentes
X = df.drop("Price", axis=1) # colunas de alimento para o modelo de AI
y = df["Price"] # coluna para o modelo prever

# Separar os dados em treinamento e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% para testes
                                                    random_state=42) # a divisão vai ser a mesma sempre que rodar

# Transformar os valores nulos
from sklearn.impute import SimpleImputer # substitui valores ausentes com uma estratégia definida ('missing', 4, mean)
from sklearn.compose import ColumnTransformer # permite aplicar transformacões diferentes para diferentes colunas

# Definir as colunas por tipos para aplicar tratamentos específicos
colunas_categoricas = ["Make", "Colour"]
coluna_door = ["Doors"]
colunas_numericas = ["Odometer (KM)"]

# Criar os imputadores
valor_substituto_categorica = SimpleImputer(strategy="constant", fill_value="missing") # usado para substituir 'missing' em colunas categóricas
valor_substituto_door = SimpleImputer(strategy="constant", fill_value=4) # usado para a coluna 'Doors'
valor_substituto_numericos = SimpleImputer(strategy="mean") # usado para colunas contínuas (numéricas)

### Criar o imputer (aplica diferentes técnicas em diferentes colunas)
# O ColumnTransformer permite processar diferentes grupos de colunas com técnicas diferentes
imputer = ColumnTransformer([
    ("valor_substituto_categorica", valor_substituto_categorica, colunas_categoricas),
    ("valor_substituto_door", valor_substituto_door, coluna_door),
    ("valor_substituto_numericos", valor_substituto_numericos, colunas_numericas)
])

### Imputer é um ColumnTransformer que contém vários SimpleImputer
# Preencher os valores ausentes (coluna por coluna) usando a estratégia definida ('missing', 4, mean)
X_train_transformed = imputer.fit_transform(X_train)
X_test_transformed = imputer.transform(X_test) # volta um array sem os nomes das colunas

# Transformar as colunas categoricas em numericas
from sklearn.preprocessing import OneHotEncoder # codificador para transformar categorias em números

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
X_train_final = transformer.fit_transform(X_train_df)
X_test_final = transformer.transform(X_test_df)

# Ajustar um modelo para treinamento
np.random.seed(42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

# Instanciar e testar o modelo
training = model.fit(X_train_final, y_train)

# Checar a pontuação do modelo com o conjunto de teste
test = model.score(X_test_final, y_test)

print(test)