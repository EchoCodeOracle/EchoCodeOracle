# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:55 2023

@author: ARK00
"""

import pandas as pd
df=pd.read_csv("bank-full.csv")
df
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('bank-full.csv')
data
data.head
len(data)
list(data)
if 'y' in data.columns:

    X = data.drop('y', axis=1)
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
else:
    print("Column 'y' not found in the dataset.")



