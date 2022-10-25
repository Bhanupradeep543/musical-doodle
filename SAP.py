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
        data = pd.DataFrame(pd.read_excel(data_file)
        data.to_csv("sap.csv", index = None, header=True)
        data=pd.DataFrame(pd.read_csv("sap.csv"))
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.subheader('total data points')
        st.dataframe(data.shape)


