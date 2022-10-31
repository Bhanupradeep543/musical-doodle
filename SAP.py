#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pickle
from sklearn import preprocessing
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import streamlit as st
import io
# choosing the image for application background directly from web URL
st.markdown(f"""<style>.stApp {{                        
             background: url("https://www.intouch-quality.com/hubfs/quality-defects-ft-lg.jpg");
             background-size: cover}}
         </style>""",unsafe_allow_html=True)
st.write("""# SEIL SAP Notification Dashboard """) # Tittle addition
st.subheader("Select the Options Below")

# adding upload button for giving test data input to model
data_file = st.file_uploader("Upload SAP EXCEL file",type=['xlsx'])
if st.button("Process"):
    if data_file is not None:
        data = pd.DataFrame(pd.read_excel(data_file))
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.subheader('Total notifications')
        st.write(data.shape[0])
        x=data['Order'].isnull().sum()
        y=data.shape[0]
        st.subheader("% of permits issued against notifications")
        st.write(round((((y-x)/y)*100),2))
        st.subheader("Priority wise notifications")
        st.write(data['Priority'].value_counts())
        st.subheader("Max. notifications Reported by")
        st.write(data['Reported by'].value_counts().head(10))
        st.subheader("Max. notifications Planner group wise")
        st.write(data['Planner group'].value_counts().head())
        st.subheader("Max. notifications Department wise")
        st.write(data['Main WorkCtr'].value_counts().head())
        st.subheader("User status of notification")
        st.write(data['User status'].value_counts().head())
        st.subheader("Repeated notifications ")
        a=data['Functional Loc.'].value_counts().head(100)
        st.write(a.iloc[1])

     

