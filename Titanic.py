# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:48:00 2020

@author: bhanu teja
"""


import pandas as pd
ds=pd.read_csv("c:/pyWork/pyData/titanic.csv")
ds.drop(["Passenger_id","name","ticket"],axis=1,inplace=True)
s_dummies=pd.get_dummies(ds["sex"],drop_first=True,)
pclass_dummies=pd.get_dummies(ds["pclass"],drop_first=True)
embarked_dummies = pd.get_dummies(ds["embarked"],drop_first=True)
import numpy as np
ds=ds.drop(["sex","pclass","embarked"],axis=1)
ds=pd.concat([ds,s_dummies,pclass_dummies,embarked_dummies],axis=1)
ds.info()
ds["fare"]=pd.to_numeric(ds.fare,errors="coerce")
ds["age"]=pd.to_numeric(ds.age,errors="coerce")
ds.isnull().sum()
import seaborn as sn
sn.heatmap(ds.isnull(),yticklabels=False)
ds.dropna(subset=["fare"],inplace=True)
ds.isnull().sum()
ds["age"].fillna(ds["age"].mean(),inplace=True)
ds.isnull().sum()
from sklearn.model_selection import train_test_split
x=ds.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
y=ds.iloc[:,0].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)   ### accuracy =  (165+84)\(165+84+30+44) = 77.5%


### backward elimination
import statsmodels.api as sms
x_1=np.append(arr=np.ones((1291,1)).astype(int),values=x,axis=1)
regressor_ols=sms.OLS(endog=y,exog=x_1).fit()
regressor_ols.summary()
x_1_1=x_1[:,[0,1,2,4,5,6,7,8,9,10]]
regressor_ols=sms.OLS(endog=y,exog=x_1_1).fit()
regressor_ols.summary()

x_1_1=x_1[:,[0,1,2,4,5,6,7,9,10]]
regressor_ols=sms.OLS(endog=y,exog=x_1_1).fit()
regressor_ols.summary()

x_1_1=x_1[:,[0,1,2,5,6,7,9,10]]
regressor_ols=sms.OLS(endog=y,exog=x_1_1).fit()
regressor_ols.summary()

x_1_1=x_1[:,[0,1,2,5,6,7,10]]
regressor_ols=sms.OLS(endog=y,exog=x_1_1).fit()
regressor_ols.summary()

### "sibsip","sex","embarked","age","pclass" are the most significant ones..








