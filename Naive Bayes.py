#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()

dir(data)

df = pd.DataFrame(data.data,columns=data.feature_names)
df.head()

df['target'] = data.target
df[50:70]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=100)

from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(X_train,y_train)

model.score(X_test,y_test)

mn = MultinomialNB()
mn.fit(X_train,y_train)
mn.score(X_test,y_test)



