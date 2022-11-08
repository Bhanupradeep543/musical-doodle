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
st.subheader("Select the date range for notifications") 
d = st.date_input("From", )
e = st.date_input("TO", )
# adding upload button for giving test data input to model
data_file = st.file_uploader("Upload SAP EXCEL file",type=['xlsx'])
if st.button("Process"):
    if data_file is not None:
        data = pd.DataFrame(pd.read_excel(data_file))
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.subheader('Total notifications')
        st.write(data.shape[0])
        st.subheader("Max. notifications Reported by")
        st.bar_chart(data['Reported by'].value_counts().head(10))
        st.subheader("Max. notifications Planner group wise")
        st.bar_chart(data['Planner group'].value_counts().head(7))
        st.subheader("User status of notification")
        st.write(data['User status'].value_counts().head())
        st.subheader("Repeated notifications ")
        st.write("File consists of TOP 300 notifications with same functional location")
        a=data.iloc[:,13].value_counts().head(300)
        st.write(a)
        st.write(data['Created On'][2])
        def convert_df(df):
          return df.to_csv().encode('utf-8')
        csv = convert_df(a) # calling the function to convert the output file into CSV
        #adding a download button to download csv file
        st.download_button(label="Download",data=csv,file_name='Repeated notifications.csv',mime='text/csv')


     

