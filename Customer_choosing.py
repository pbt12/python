# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:17:58 2020

@author: bhanu teja
"""


import pandas as pd
ds=pd.read_csv("Purchase_History.csv")
del ds["User ID"]
g_dummies = pd.get_dummies(ds["Gender"],drop_first=True)
ds_1=pd.concat([ds,g_dummies],axis=1)
ds_1=ds_1.drop(["Gender"],axis=1)
x=ds_1.iloc[:,[0,1,3]]
y=ds_1.iloc[:,2]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy",max_depth = 3,min_samples_leaf=5)
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)   ### accuracy = 55 + 21 / 55+3+1+21 = 76 / 80
print(7600/80)  ## accuracy = 95%
from sklearn import tree
tree.plot_tree(dtc)
cn=["0","1"]
tree.plot_tree(dtc,class_names = cn,filled=True)
import matplotlib.pyplot as plt
fig=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
tree.plot_tree(dtc,class_names = cn,filled=True)
