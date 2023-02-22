import streamlit as st
import os
import sys
from utils import recommend
from PIL import Image
import pickle
import requests
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import glob
import io
import codecs

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
data_folder = (PROJECT_ROOT + "/" + "data")
body_folder = (PROJECT_ROOT + "/" + "body")

saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
content_based_supervised_data = (data_folder + "/" + "processed" + "/" + "content_based_supervised")
images = (PROJECT_ROOT + "/" + "images")
cover_images = (images + "/" + "Cover_images")



def it_is_about():
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Anime Recommendation systems based on Data from MyAnimeList**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("The goal of this project is that according to the user's anime viewing history we can recommend a list of anime that suits their tastes.\nIn order to do this we are going to create 3 types or recommendation system")



    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Unsupervised Collaborative Filtering based on ratings Using k-Nearest Neighbors (kNN)</p>', unsafe_allow_html=True)
    with st.expander("See explanation and process."):
        with open(body_folder + "/" + "un_based_feat.markdown",'r', encoding='utf-8') as f:
            st.markdown(f.read(), unsafe_allow_html=True)    


    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Unsupervised content based recommendation system</p>', unsafe_allow_html=True)
    with st.expander("See explanation and process."):
        path_to_html = body_folder + "/" + "unsupervised_content_based_filtered_filtered.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=10600)



    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Supervised Collaborative Filtering based on ratings Using SVD method</p>', unsafe_allow_html=True)
    with st.expander("See explanation and process."):
        st.write("Explanation body goes here")