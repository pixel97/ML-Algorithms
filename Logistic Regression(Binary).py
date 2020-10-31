#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv(r"C:\My Projects\ML Algorithms\Logistic Regression(Binary)\insurance_info.csv")
df.head()


# In[12]:


plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


# In[13]:


df.shape


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.2)


# In[19]:


X_test


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


model = LogisticRegression()


# In[22]:


model.fit(X_train,y_train)


# In[23]:


model.predict(X_test)


# In[25]:


model.score(X_test,y_test)


# In[ ]:




