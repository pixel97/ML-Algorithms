#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[7]:


data = load_iris()


# In[9]:


dir(data)


# In[45]:


data.data[0]


# In[16]:


data.data[0]


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[21]:


model = LogisticRegression()


# In[31]:


X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2)


# In[32]:


model.fit(X_train,y_train)


# In[33]:


model.predict(data.data[0:5])


# In[34]:


model.score(X_test,y_test)


# In[35]:


y_pred =  model.predict(X_test)


# In[36]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[48]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




