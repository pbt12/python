#!/usr/bin/env python
# coding: utf-8

# # GRIP TASK #1
#  
# # P.BHANU TEJA
# 
# # SCORE PREDICTION USING TOTAL HOURS STUDIED BY A STUDENT ( SUPERVISED MACHINE LEARNING )
# 
# # February batch-2021

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importing dataset using pandas

# In[2]:


url = " https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv "
df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df


# # now we will see wether we need to use linear regression or multiple regression
# 
# obviously it will be a linear regression as we have only two feautes and one should be our required one 
# so we will go with linear regression algorithm

# In[3]:


df.any()


# so no null values 

# In[7]:


df.plot(x='Hours',y='Scores',style='o')
plt.xlabel("Hours")
plt.ylabel('Scores')
plt.title(" Hours Vs Scores")


# fine linear regression confirmed as hours vs score seems to be in a near linear relation

# # testing and traing the model

# In[25]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


regressor=LinearRegression()


# In[34]:


regressor.fit(x_train,y_train)


# In[38]:


y_pred=regressor.coef_*x + regressor.intercept_
plt.scatter(x_train,y_train)
plt.plot(x,y_pred)


# In[39]:


print(x_test)


# In[40]:


y_pred=regressor.predict(x_test)


# In[41]:


print(y_pred)


# In[42]:


df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
df


# dataframe showing the predicted values by our model and actual values 

# # predicting tha score for 9.25 hrs of studying

# In[44]:


hours = 9.25
pred_score=regressor.predict([[hours]])
print(pred_score)


# so, our model predicts that the student scores 93.691 marks if he/she study 9.25 hours

# # calculating mean absolute and mean squared errors of our predicted and actaual values

# In[45]:


from sklearn import metrics
y_pred = regressor.predict(x_test)


# In[46]:


print("Mean Absolute Error = ",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error = ",metrics.mean_squared_error(y_test,y_pred))


# In[ ]:




