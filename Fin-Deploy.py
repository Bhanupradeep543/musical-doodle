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


# In[25]:


train_data=train_data.drop(columns=['recorded_by','id','quantity_group','num_private','payment_type','extraction_type_group','quality_group','source_type','waterpoint_type_group','date_recorded','region','region_code','extraction_type','scheme_management','installer','amount_tsh','management_group']) 
train_data.shape


# In[26]:


train_data=train_data.drop(columns=['wpt_name','ward','subvillage','lga'])
train_data.shape


# In[27]:


label_encoder = preprocessing.LabelEncoder()
train_data['status_group']= label_encoder.fit_transform(train_data['status_group']) 
print(train_data['status_group'].value_counts())


# In[28]:


mu, sigma = 0, 0.05 # mean and % of noise for adding noise


# In[29]:


encoder= ce.TargetEncoder(cols=['permit'],min_samples_leaf=40, smoothing=10) # Target encoding parameters
train_data['permit']= encoder.fit_transform(train_data['permit'],train_data['status_group']) # fitting model with the categorical column and target variable
noise = np.random.normal(mu, sigma, [59400])
train_data['permit']=noise+train_data['permit']


# TargetEncoding method is chosen to avoid the sparsity and avoid the higher dimensionality and also to avoid the models overfitting gaussian noise of 5% is added to all encoded columns of tragetencoder.

# In[30]:


encoder= ce.TargetEncoder(cols=['public_meeting'],min_samples_leaf=40, smoothing=10)
train_data['public_meeting']= encoder.fit_transform(train_data['public_meeting'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['public_meeting']=noise+train_data['public_meeting']


# In[31]:


encoder= ce.TargetEncoder(cols=['water_quality'],min_samples_leaf=40, smoothing=10)
train_data['water_quality']= encoder.fit_transform(train_data['water_quality'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['water_quality']=noise+train_data['water_quality']


# In[32]:


encoder= ce.TargetEncoder(cols=['quantity'],min_samples_leaf=40, smoothing=10)
train_data['quantity']= encoder.fit_transform(train_data['quantity'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['quantity']=noise+train_data['quantity']


# In[33]:


encoder= ce.TargetEncoder(cols=['waterpoint_type'],min_samples_leaf=40, smoothing=10)
train_data['waterpoint_type']= encoder.fit_transform(train_data['waterpoint_type'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['waterpoint_type']=noise+train_data['waterpoint_type']


# In[34]:


encoder= ce.TargetEncoder(cols=['source_class'],min_samples_leaf=40, smoothing=10)
train_data['source_class']= encoder.fit_transform(train_data['source_class'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['source_class']=noise+train_data['source_class']


# In[35]:


encoder= ce.TargetEncoder(cols=['source'],min_samples_leaf=40, smoothing=10)
train_data['source']= encoder.fit_transform(train_data['source'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['source']=noise+train_data['source']


# In[36]:


encoder= ce.TargetEncoder(cols=['payment'],min_samples_leaf=40, smoothing=10)
train_data['payment']= encoder.fit_transform(train_data['payment'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['payment']=noise+train_data['payment']


# In[37]:


encoder= ce.TargetEncoder(cols=['management'],min_samples_leaf=40, smoothing=10)
train_data['management']= encoder.fit_transform(train_data['management'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['management']=noise+train_data['management']


# In[38]:


encoder= ce.TargetEncoder(cols=['extraction_type_class'],min_samples_leaf=40, smoothing=10)
train_data['extraction_type_class']= encoder.fit_transform(train_data['extraction_type_class'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['extraction_type_class']=noise+train_data['extraction_type_class']


# In[39]:


encoder= ce.TargetEncoder(cols=['basin'],min_samples_leaf=40, smoothing=10)
train_data['basin']= encoder.fit_transform(train_data['basin'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['basin']=noise+train_data['basin']


# In[40]:


encoder= ce.TargetEncoder(cols=['funder'],min_samples_leaf=40, smoothing=10)
train_data['funder']= encoder.fit_transform(train_data['funder'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['funder']=noise+train_data['funder']


# In[41]:


encoder= ce.TargetEncoder(cols=['scheme_name'],min_samples_leaf=50, smoothing=10)
train_data['scheme_name']=encoder.fit_transform(train_data['scheme_name'],train_data['status_group'])
noise = np.random.normal(mu, sigma, [59400])
train_data['scheme_name']=noise+train_data['scheme_name']


# In[42]:


train_data['population'].fillna(0)
train_data['population']=train_data['population'].replace(0,train_data['population'].median())


# assigining the missing values with median of the data for the population feature is appropriate. We cannot assign population with zero, as population in a region cannot be zero.

# In[43]:


train_data['construction_year']=train_data['construction_year'].replace(0,train_data['construction_year'].median())
train_data['construction_year'].describe()


# In[44]:


data=train_data


# In[51]:


data.shape


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

