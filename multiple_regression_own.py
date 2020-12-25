# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 20:00:59 2020

@author: bhanu teja
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ds=pd.read_csv("C:/Users/bhanu teja/OneDrive/Desktop/stud_reg_2.csv")

x=ds.iloc[:,:-1]
y=ds.iloc[:,2]
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.3,random_state=0)   ###random state= 0 has high acciracy

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(regressor.coef_)
print(regressor.intercept_)
print(r2_score(y_test,y_pred))






