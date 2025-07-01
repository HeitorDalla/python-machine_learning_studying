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
X = df.drop("Price", axis=1)
y = df["Price"]

# Separar os dados em treinamento e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42)

# Transformar os valores nulos com sklearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Definir as colunas por tipos
cat_features = ["Make", "Colour"]
door_feature = ["Doors"]
num_features = ["Odometer (KM)"]

# Criar os imputadores
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
door_imputer = SimpleImputer(strategy="constant", fill_value=4)
num_imputer = SimpleImputer(strategy="mean")

# Criar o imputer
imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_feature),
    ("num_imputer", num_imputer, num_features)
])

# Preencher os dados de treinamento e teste separadamente
X_train_transformed = imputer.fit_transform(X_train)
X_test_transformed = imputer.transform(X_test)

# Transformar as colunas categoricas em numericas
from sklearn.preprocessing import OneHotEncoder # transformar categorias em números

variaveis_categoricas = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder() # cria um codificador de texto para números binários, criando uma coluna com 0 ou 1

transformer = ColumnTransformer([('one_hot', # nome da transformação
                                 one_hot, 
                                 variaveis_categoricas)], # lista como o nome das variáveis categóricas
                                 remainder='passthrough') # define o que fazer com as colunas que não estão nas lista de transformação, no caso, mantém no conjunto de dados

# Como os dados foram transformados em arrays pelo imputer, precisamos recriar os DataFrames
# com os mesmos nomes de colunas para aplicar o OneHotEncoder baseado nos nomes
X_train_df = pd.DataFrame(X_train_transformed, columns=cat_features + door_feature + num_features)
X_test_df = pd.DataFrame(X_test_transformed, columns=cat_features + door_feature + num_features)

# Aplicar a codificação
X_train_final = transformer.fit_transform(X_train_df)
X_test_final = transformer.transform(X_test_df)

# Ajustar um modelo para treinamento
np.random.seed(42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

trainning = model.fit(X_train_final, y_train)
test = model.score(X_test_final, y_test)

print(trainning)
print(test)