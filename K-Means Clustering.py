#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


iris = load_iris()

dir(iris)

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df['target'] = iris.target
df.head(10)

inputs = df.drop(['sepal length (cm)','sepal width (cm)','target'],axis='columns')
inputs.head()

plt.scatter(inputs['petal length (cm)'],df['petal width (cm)'])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

model = KMeans(n_clusters=3)
y = model.fit_predict(inputs)
y

inputs['group'] = y
inputs.head()

df1 = inputs[inputs.group==0]
df2 = inputs[inputs.group==1]
df3 = inputs[inputs.group==2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='green')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

model.cluster_centers_

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='green')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='yellow',marker='*',label='centroid')
plt.xlabel('Petal length(cm)')
plt.ylabel('Petal width (cm)')
plt.legend()

# Elbow method

sse = []
krange = range(1,10)
for i in krange:
    model = KMeans(n_clusters=i)
    model.fit(inputs)
    sse.append(model.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(krange,sse)





