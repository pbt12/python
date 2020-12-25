# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:13:13 2020

@author: bhanu teja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/bhanu teja/OneDrive/Desktop/stud_reg.csv")

df.isnull().sum()
df1=df["PLACE_RATE"].fillna(df["PLACE_RATE"].mean(),inplace=False)

df.to_csv("C:/Users/bhanu teja/OneDrive/Desktop/stud_reg_edited.csv")

values=[2,54,65,52,5,5.2,2.8]
plt.plot(values)
plt.show()

sales1=[2,5,7,8,6,4,2,4,6,6]
sales2=[8,5,4,8,6,2,1,5,6,5]

lc_1=plt.plot(sales1,range(1,11))

lc_2=plt.plot(sales2,range(1,11))
plt.title("monthly sales of 2016 and 2017")
plt.xlabel("month")
plt.ylabel("sales")
plt.legend(["year 2016","year 2017"],loc=4)
plt.show()

values=[60,80,90,55,10,30]
colors=["b","g","r","c","m","y",]
labels = ["Us","uk","india","germany","australia","s.korea"]

explode = (0.5,0,0,0,0,0)

plt.pie(values,colors=colors,labels=values,explode=explode,counterclock=False,shadow=True)


val = [28,26,25,26,27,29,32]
colors=["b","g","r","c","m","y","PuRd"]
labels=["us","india","uk","n.korea","australia","sweden","france"]
explode = (0.5,0,0,0,0,0,0)

plt.pie(val,colors=colors,labels=labels,explode=explode,counterclock=False,shadow=True)





