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

    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Anime Recommendation systems based on Data from MyAnimeList</p>', unsafe_allow_html=True)
    st.write("The goal of this project is that according to the user's anime viewing history we can recommend a list of anime that suits their tastes.\nIn order to do this we are going to create 3 types or recommendation system")

    with open(body_folder + "/" + "about_text.md",'r', encoding='utf-8') as f:
        st.markdown(f.read(), unsafe_allow_html=True)