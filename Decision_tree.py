#!/usr/bin/env python
# coding: utf-8

import pandas as pd

data = pd.read_csv(r"C:\My Projects\ML Algorithms\Decision Tree\titanic.csv")

data.head()

data.Age.isnull().value_counts()

data['Age'] = data['Age'].fillna(data['Age'].mean())
data.Age.isnull().value_counts()


inputs = data.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')


inputs.head()


target = data['Survived']
target.head()

from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()

inputs['le_sex'] = le_sex.fit_transform(inputs['Sex']) 

inputs.head()


inputs = inputs.drop(['Sex'],axis='columns')

inputs.head()

from sklearn.model_selection import train_test_split
from sklearn import tree

X_train,X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)



model = tree.DecisionTreeClassifier()

model.fit(X_train,y_train)


model.score(X_test,y_test)





