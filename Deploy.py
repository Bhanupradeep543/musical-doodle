#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from mlxtend.classifier import StackingClassifier
data=pd.read_csv("phase2_output.csv")


# In[2]:


data.shape


# In[3]:


print(data['status_group'].value_counts())


# converted the target variable also into numerical

# From the above dataset information, all the features are converted into numerical for modelling

# In[4]:


data=data.drop(columns=['Unnamed: 0']) 


# it is a serial number column after exporting data into CSV file

# In[5]:


X=data.drop(columns=['status_group'])
y=data['status_group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Splitting the data randomly Train data: 80% and test data: 20%

# In[11]:


#choosing the hyperparameters based on dataset
XGB = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree',num_class = 3,
                    eta = .1,max_depth = 10, colsample_bytree = .4, 
                    learning_rate = 0.1,max_delta_step=1)
#fitting the train data input variables and target variables
XGB.fit(X_train, y_train)
# predicting the target varible from input variable of train data
y_pred = XGB.predict(X_train)
# predicting the target varible from input variable of test data
y_pred_test = XGB.predict(X_test)
print("F1 Score:")
F1_train=round(f1_score(y_train, y_pred,average='micro'),2)
F1_test=round(f1_score(y_test, y_pred_test,average='micro'),2)
print("TRAIN dataset :",F1_train)
print("TEST dataset  :",F1_test)
summary.loc[len(summary.index)] = ['XGboost',F1_train,F1_test] 

