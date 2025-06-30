import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # embaralhar os dados

df = pd.read_csv("data/car-sales-extended.csv")

X = df.drop('Price', axis=1)
y = df['Price']

# Verificar dataframe - Existem dois campos que sao objetos, tem que transformar em numéricos
print(df.info())

# Transformar as categorias em números
from sklearn.preprocessing import OneHotEncoder # transformar categorias em números
from sklearn.compose import ColumnTransformer

variaveis_categoricas = ['Make', 'Colour', 'Doors'] # Doors entra por ser uma variável categorizadora numérica
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot',
                                 one_hot,
                                 variaveis_categoricas)],
                                 remainder='passthrough')

transformed_X = transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformed_X, # colocando o dataframe numérico apenas
                                                    y, 
                                                    test_size=0.2)

# Escolher o modelo e treina-lo
from sklearn.ensemble import RandomForestRegressor # modelo de regressão

model = RandomForestRegressor()

trainning = model.fit(X_train, y_train)
test = model.score(X_test, y_test)

print(trainning)
print(test)