# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:37:28 2023

@author: ARK00
"""


===============================================================================
import pandas as pd

data = pd.read_csv('50_Startups.csv')

X = data.drop('Profit', axis=1)
y = data['Profit']

numerical_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']

numeric_transformer = Pipeline(steps=[('num', 'passthrough')])

from sklearn.compose import ColumnTransformer
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

from sklearn.pipeline import Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
# Calculate R^2 value
r2 = r2_score(y_test, y_pred)
print(f'R^2 Value: {r2}')

===============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

corolla = pd.read_csv('ToyotaCorolla.csv', encoding='latin1')
# Select the specified columns
corolla = corolla[["Price", "Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]]

X = corolla.drop('Price', axis=1)
y = corolla['Price']

numerical_features = ['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']
categorical_features = []  # No categorical features in this subset

numeric_transformer = Pipeline(steps=[('num', 'passthrough')])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate R^2 value
r2 = r2_score(y_test, y_pred)
print(f'R^2 Value: {r2}')







