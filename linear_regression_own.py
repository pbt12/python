# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:56:47 2020

@author: bhanu teja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("C:/Users/bhanu teja/OneDrive/Desktop/stud_reg.csv")
df.describe()
df.isnull().sum()
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
r2_score(y_test,y_pred)

print(regressor.coef_)
print(regressor.intercept_)







