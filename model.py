#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn import preprocessing
from scipy import stats
import category_encoders as ce
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import category_encoders as ce
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from mlxtend.classifier import StackingClassifier
train_data=pd.read_csv("tdata_proj_82.csv") # loading the train data file in pandas
train_data.shape # included train data labels along with train data values


# In[45]:


X=data.drop(columns=['status_group'])
y=data['status_group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


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
op=pd.DataFrame(y_pred_test)
op=op.replace(to_replace=0,value="functional")
op=op.replace(to_replace=1,value="non functional")
op=op.replace(to_replace=2,value="functional needs repair")
print(op)

