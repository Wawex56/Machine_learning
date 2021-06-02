#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 02:43:04 2021

@author: wawex
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


dataset = pd.read_csv('maaslar.csv')
# data frame
x = dataset.iloc[:,1:2]
y = dataset.iloc[:,2:]

#numpy array 
X = x.values
Y = y.values


# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# polynomial regression
# 2 degree polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')

# 4 degree polynomial
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

# visualization
plt.scatter(X,Y, color='red')
plt.plot(x,lin_reg.predict(X), color='blue')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'black')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

# predict
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


# scaling of data
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_scaled = sc1.fit_transform(X)

sc2=StandardScaler()
y_scaled = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)

plt.scatter(x_scaled, y_scaled)
plt.plot(x_scaled,svr_reg.predict(x_scaled))

svr_reg.predict([[11]])
svr_reg.predict([[6.6]])
