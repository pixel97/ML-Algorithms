#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import math


# In[4]:


data = pd.read_csv(r"C:\Users\dkjai\Documents\houseprices_multi.csv")
data.head()


# In[6]:


# Handle the missing bedroom value by taking median


# In[10]:


median_bedrooms = math.floor(data.bedroom.median())
median_bedrooms


# In[14]:


data.bedroom = data.bedroom.fillna(median_bedrooms)


# In[15]:


data.head()


# In[17]:


model = linear_model.LinearRegression()
model.fit(data[['area','bedroom','age']],data.price)


# In[18]:


model.coef_


# In[19]:


model.intercept_


# In[20]:


# predict the price for 3000 sqft, 4 bedrooms, 40 yrs old


# In[24]:


model.predict([[3000,4,40]])


# In[ ]:




