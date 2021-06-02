#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:13:47 2021

@author: wawex
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


dataset = pd.read_csv('satislar.csv')
print(dataset)

months = dataset[['Aylar']]
print(months)

sales = dataset[['Satislar']]
print(sales)

#verilerin egitim ve test icin bolunmesi 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months,sales, test_size=0.33, random_state=0)

# (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

pred = lr.predict(x_test)

x_train = x_train.sort_index()
t_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title('sales by month')
plt.xlabel("months")
plt.ylabel("sales")
