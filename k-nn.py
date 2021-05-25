#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 02:01:42 2021

@author: wawex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


veriler = pd.read_csv('veriler.csv')
print(veriler)
x = veriler.iloc[:,1:4].values # bagimsiz degiskenler
y = veriler.iloc[:,4:].values # bagimli degiskenler


# verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# logistic regresyon
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_k_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_k_pred)
print(cm) 