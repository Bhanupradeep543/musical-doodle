import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 
import pandas as pd
#title
st.title("Extract Text from Images")
#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])
@st.cache
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader 
reader = load_model() #load model
if image is not None:
    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 wait for a while! "):
        result = reader.readtext(np.array(input_image))
        
        result_text = pd.DataFrame() #empty list for results
        result_text=result_text.append(result)
        result_text=result_text.drop(details.columns[[0,2]],axis = 1)
        st.write(result_text)
    #st.success("Here you go!")
    st.balloons()
    def convert_df(df):
          return df.to_csv().encode('utf-8')
    csv = convert_df(result_text) # calling the function to convert the output file into CSV
    #adding a download button to download csv file
    st.download_button(label="Download",data=csv,file_name='Generator temp.csv',mime='text/csv')
else:
    st.write("Upload an Image")

