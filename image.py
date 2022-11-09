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

    with st.spinner("🤖 AI is at Work! "):
        result = reader.readtext(np.array(input_image))
        
        result_text = pd.DataFrame() #empty list for results
        result_text=result_text.append(result)

        st.write(result_text)
    #st.success("Here you go!")
    st.balloons()
else:
    st.write("Upload an Image")

