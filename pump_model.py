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
import xgboost as xgb
from xgboost import XGBClassifier
import streamlit as st


# In[ ]:
st.write("""Predicting Faulty Pump in Tanzania waterpoints dataset""")
st.subheader("Waterpoint Input parameters")
data_file = st.file_uploader("Upload CSV",type=['csv'])
if st.button("Process"):
    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        st.dataframe(df)
raw_data=pd.read_csv("https://drivendata-prod.s3.amazonaws.com/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20221007%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221007T103648Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=93ed9259152e9331b2bd4f0d34506025f5231e25bd6c932fa12b4fc6707d816d") 
target=pd.read_csv("https://drivendata-prod.s3.amazonaws.com/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20221007%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221007T103648Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3eff4f9628acf58a18081993416286cbcef9c2706b207adf441fff26172b5734")

# In[ ]:
def datacleaning(train_data):
    train_data=pd.concat((train_data,target),axis=1)
    label_encoder = preprocessing.LabelEncoder()
    train_data['status_group']= label_encoder.fit_transform(train_data['status_group']) 
    train_data=train_data.drop(columns=['recorded_by','id','quantity_group','num_private','payment_type','extraction_type_group','quality_group','source_type','waterpoint_type_group','date_recorded','region','region_code','extraction_type','scheme_management','installer','amount_tsh','management_group',
                                   'wpt_name','ward','subvillage','lga']) 
    mu, sigma = 0, 0.05 # mean and % of noise for adding noise
    encoder= ce.TargetEncoder(cols=['permit'],min_samples_leaf=40, smoothing=10) # Target encoding parameters
    train_data['permit']= encoder.fit_transform(train_data['permit'],train_data['status_group']) # fitting model with the categorical column and target variable
    noise = np.random.normal(mu, sigma, [59400])
    train_data['permit']=noise+train_data['permit']
    encoder= ce.TargetEncoder(cols=['public_meeting'],min_samples_leaf=40, smoothing=10)
    train_data['public_meeting']= encoder.fit_transform(train_data['public_meeting'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['public_meeting']=noise+train_data['public_meeting']
    encoder= ce.TargetEncoder(cols=['water_quality'],min_samples_leaf=40, smoothing=10)
    train_data['water_quality']= encoder.fit_transform(train_data['water_quality'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['water_quality']=noise+train_data['water_quality']
    encoder= ce.TargetEncoder(cols=['quantity'],min_samples_leaf=40, smoothing=10)
    train_data['quantity']= encoder.fit_transform(train_data['quantity'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['quantity']=noise+train_data['quantity']
    encoder= ce.TargetEncoder(cols=['waterpoint_type'],min_samples_leaf=40, smoothing=10)
    train_data['waterpoint_type']= encoder.fit_transform(train_data['waterpoint_type'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['waterpoint_type']=noise+train_data['waterpoint_type']
    encoder= ce.TargetEncoder(cols=['source_class'],min_samples_leaf=40, smoothing=10)
    train_data['source_class']= encoder.fit_transform(train_data['source_class'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['source_class']=noise+train_data['source_class']
    encoder= ce.TargetEncoder(cols=['source'],min_samples_leaf=40, smoothing=10)
    train_data['source']= encoder.fit_transform(train_data['source'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['source']=noise+train_data['source']
    encoder= ce.TargetEncoder(cols=['payment'],min_samples_leaf=40, smoothing=10)
    train_data['payment']= encoder.fit_transform(train_data['payment'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['payment']=noise+train_data['payment']
    encoder= ce.TargetEncoder(cols=['management'],min_samples_leaf=40, smoothing=10)
    train_data['management']= encoder.fit_transform(train_data['management'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['management']=noise+train_data['management']
    encoder= ce.TargetEncoder(cols=['extraction_type_class'],min_samples_leaf=40, smoothing=10)
    train_data['extraction_type_class']= encoder.fit_transform(train_data['extraction_type_class'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['extraction_type_class']=noise+train_data['extraction_type_class']
    encoder= ce.TargetEncoder(cols=['basin'],min_samples_leaf=40, smoothing=10)
    train_data['basin']= encoder.fit_transform(train_data['basin'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['basin']=noise+train_data['basin']
    encoder= ce.TargetEncoder(cols=['funder'],min_samples_leaf=40, smoothing=10)
    train_data['funder']= encoder.fit_transform(train_data['funder'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['funder']=noise+train_data['funder']
    encoder= ce.TargetEncoder(cols=['scheme_name'],min_samples_leaf=50, smoothing=10)
    train_data['scheme_name']=encoder.fit_transform(train_data['scheme_name'],train_data['status_group'])
    noise = np.random.normal(mu, sigma, [59400])
    train_data['scheme_name']=noise+train_data['scheme_name']
    train_data['population'].fillna(0)
    train_data['population']=train_data['population'].replace(0,train_data['population'].median())
    train_data['construction_year']=train_data['construction_year'].replace(0,train_data['construction_year'].median())
    cleandata=train_data
    return cleandata


# In[43]:


data=datacleaning(raw_data)
test=datacleaning(df)
x=data.drop(columns=['status_group'])
y=data['status_group']


# In[46]:


#choosing the hyperparameters based on dataset
XGB = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree',num_class = 3,
                    eta = .1,max_depth = 10, colsample_bytree = .4, 
                    learning_rate = 0.1,max_delta_step=1)
#fitting the train data input variables and target variables
XGB.fit(x, y)
# predicting the target varible from input variable of train data
test=test.drop(columns=['status_group'])
prediction = XGB.predict(test)
prediction=pd.DataFrame(prediction)


st.subheader('Prediction')
st.write(predicition)



