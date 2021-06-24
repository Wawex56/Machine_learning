#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 00:16:28 2021

@author: wawex
"""

import pandas as pd

dataset = pd.read_excel('Churn_Modelling.xlsx')
#print(dataset)

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values  

# Categoric -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer(["ohe", OneHotEncoder(dtype=float),[1]], remainder = "passthrough")

#X = ohe.fit_transform(X)
#X = X[:,1:]

#splitting data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#scaling of data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# neural network 
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() 
classifier.add(Dense(6, activation = 'relu', input_dim = 10)) 
classifier.add(Dense(6, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train,y_train, epochs=50)

# evalute the model
scores = classifier.evaluate(X_train,y_train)
print("\n%s: %.2f%%" % (classifier.metrics_names[1],scores[1]*100))

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test,y_pred)
print(cm)

from ann_visualizer.visualize import ann_viz;
ann_viz(classifier, title="My first neural network")
