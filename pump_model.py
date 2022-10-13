#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
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
import io
# choosing the image for application background directly from web URL
st.markdown(f"""<style>.stApp {{                        
             background: url("https://image.shutterstock.com/image-vector/artificial-intelligence-concept-technology-background-260nw-1111962779.jpg");
             background-size: cover
         }}
         </style>""",unsafe_allow_html=True)
st.write("""# Predicting Faulty Pump in Tanzania waterpoints """) # Tittle addition
st.subheader("Dataset")
st.write("""Input dataset should have below features and datatype""")
# accessing the train datasets from web URLs
raw_data=pd.read_csv("https://drivendata-prod.s3.amazonaws.com/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20221013%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221013T103111Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=55ef17854c3d43787ebf0918408427731b786d2be8314d8c3228e4584329f52c") 
target=pd.read_csv("https://drivendata-prod.s3.amazonaws.com/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20221013%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221013T103111Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=978a46910db82acddcb6d90e0d0289cafd62a40c87c8e3216ffcad8eadabc432")
buffer = io.StringIO()
raw_data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)   # for printing the datafram information

# adding upload button for giving test data input to model
data_file = st.file_uploader("Upload CSV",type=['csv'])
st.write("After uploading the file Name Error will be rectified.")
if st.button("Process"):
    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        df = pd.read_csv(data_file)
        st.subheader('Waterpoint dataset parameters')
        st.dataframe(df)
# created a function datacleaning for working both on train and test data
def datacleaning(train_data):
    train_data=pd.concat((train_data,target),axis=1)
    label_encoder = preprocessing.LabelEncoder() # for traget encoding first we have to convert the target label to numerical
    train_data['status_group']= label_encoder.fit_transform(train_data['status_group']) 
    # removing the features using EDA analysis
    train_data=train_data.drop(columns=['recorded_by','id','quantity_group','num_private','payment_type','extraction_type_group','quality_group','source_type','waterpoint_type_group','date_recorded','region','region_code','extraction_type','scheme_management','installer','amount_tsh','management_group',
                                   'wpt_name','ward','subvillage','lga']) 
    mu, sigma = 0, 0.05 # mean and % of noise for adding noise to avoid overfitting
    encoder= ce.TargetEncoder(cols=['permit'],min_samples_leaf=40, smoothing=10) # Target encoding parameters
    train_data['permit']= encoder.fit_transform(train_data['permit'],train_data['status_group']) # fitting model with the categorical column and target variable
    noise = np.random.normal(mu, sigma, [59400]) # generating the noise values equal to rows.
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
test=datacleaning(df)
# predicting the target varible from input variable of train data
test=test.drop(columns=['status_group'])

XGB = pickle.load(open('xgbmodel_pkl','rb'))

prediction = XGB.predict(test)
# the prediction ouput is a numpy array..converting it into pandas dataframe
prediction=pd.DataFrame(prediction)
# decoding the target  labels again into categorical from numerical.
prediction=prediction.replace(to_replace=0,value="functional")
prediction=prediction.replace(to_replace=1,value="non functional")
prediction=prediction.replace(to_replace=2,value="functional needs repair")
st.subheader('Prediction')
st.write(prediction)

def convert_df(df):
    return df.to_csv().encode('utf-8')
csv = convert_df(prediction) # calling the function to convert the output file into CSV
#adding a download button to download csv file
st.download_button(label="Download data as CSV",data=csv,file_name='predicted ouput.csv',mime='text/csv')

