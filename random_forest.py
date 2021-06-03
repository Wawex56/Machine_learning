#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 18:34:08 2021

@author: wawex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


dataset = pd.read_excel('lawoffice.xlsx')
x = dataset.iloc[:,2:3]
y = dataset.iloc[:,3:,]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.55, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# logistic regression
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

# random forest for classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_rfc_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_rfc_pred)
print('RFC')
print(cm)
