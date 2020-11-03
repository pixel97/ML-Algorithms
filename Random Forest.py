#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

iris.target_names

df['target'] = iris.target
df.head()

X = df.drop('target',axis='columns')
y = df.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, y_train)


model.score(X_test, y_test)



