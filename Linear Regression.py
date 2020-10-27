#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[4]:


data = pd.read_csv(r"C:\Users\dkjai\Documents\houseprices.csv")


# In[5]:


data


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqft)')
plt.ylabel('price(US$)')
plt.scatter(data.Area,data.Price,color='red',marker='+')


# In[31]:


model = linear_model.LinearRegression()
model.fit(data[['Area']],data.Price)


# In[15]:


model.predict([[5000]])


# In[18]:


print(model.coef_)
print(model.intercept_)


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqft)')
plt.ylabel('price(US$)')
plt.scatter(data.Area,data.Price,color='red',marker='+')
plt.plot(data.Area,model.predict(data[['Area']]),color='blue')


# In[19]:


data_predict = pd.read_csv(r"C:\Users\dkjai\Documents\houseprices_predict.csv")


# In[20]:


data_predict.head()


# In[22]:


p = model.predict(data_predict)


# In[23]:


data_predict['prices'] = p


# In[27]:


data_predict.to_csv(r"C:\Users\dkjai\Documents\houseprices_predict.csv",index=False)


# In[ ]:





# In[ ]:




