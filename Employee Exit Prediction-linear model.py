#!/usr/bin/env python
# coding: utf-8

# Data - Link
# Target Column - 'left' (0/1) represents exit or not
# Data is heterogeneous in nature, so would need preprocessing before you feed them to ML models
# Build a model to predict left column
# Don't forget to spilt data in train & test subsets
# Data Wrangling & insights graphs are always an important part of data science

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
import seaborn as sns
print("Done")


# In[6]:


df=pd.read_csv('data/Predicting Employee Exit.csv')
df.head(5)


# In[4]:


df.isnull().sum()


# In[9]:


sns.lmplot(x="satisfaction_level",y="left",data=df,order=2,ci=None)


# In[11]:


X=np.array(df['satisfaction_level']).reshape(-1,1)
y=np.array(df['left']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

regr=LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_test,y_test))


# In[10]:


df.fillna(method='ffill',inplace=True)


# In[12]:


y_pred=regr.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[30]:


df.fillna(method='ffill',inplace=True)
X=np.array(df['satisfaction_level']).reshape(-1,1)
y=np.array(df['left']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

regr=LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_test,y_test))


# In[29]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse=mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
print("mae",mae)
print("mse",mse)
print("rmse",rmse)


# In[ ]:





# In[ ]:




