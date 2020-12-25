# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 20:54:25 2020

@author: bhanu teja
"""


import pandas as pd
ds=pd.read_csv("C:/pyWork/pyData/50_Startups.csv")

import numpy as np
s_dummy=pd.get_dummies(ds["State"],drop_first=True)
ds=pd.concat([ds,s_dummy],axis=1)
del ds["State"]

from sklearn.model_selection import train_test_split

x=ds.iloc[:,[0,1,2,4,5]].values
y=ds.iloc[:,3].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(regressor.coef_)
print(regressor.intercept_)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

import statsmodels.api as sm
import numpy as np
x=np.append(arr=np.ones((50,1),dtype=int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
#2
x_opt=x[:,[0,1,2,3,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
#3
x_opt=x[:,[0,1,2,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
#4
x_opt=x[:,[0,1,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
#5
x_opt=x[:,[0,1]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
#most signifecant is 1 i.e., R&D spend
ds=pd.read_csv("C:/pyWork/pyData/50_Startups.csv")
x_2=ds.iloc[:,:-4].values
y_2=ds.iloc[:,4].values
x_train2,x_test2,y_train2,y_test2=train_test_split(x_2,y_2,test_size=0.2,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train2,y_train2)
y_pred2 = regressor.predict(x_test2)
print(regressor.coef_)
print(regressor.intercept_)
r2_score(y_test,y_pred2)



