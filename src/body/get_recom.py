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
from streamlit_option_menu import option_menu
from body import un_based_rate
from body import un_based_feat
from body import sup_id
from body import about


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




def get_the_recom():
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Anime Recommendation systems based on Data from MyAnimeList**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Choose betwee the 3 types of recommendations systems to get different kind of recommendations.")

            
    selected2 = option_menu(
        None, ["Recommmend Based on ratings", "Recommmend Based on Features", "Recommmend Using user ID"], 
        icons=['123', 'list-ul', 'credit-card-2-front'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal")

    if selected2 == "Recommmend Based on ratings":
        un_based_rate.uns_bara()

    elif selected2 == "Recommmend Based on Features":
        un_based_feat.uns_feat()

    elif selected2 == "Recommmend Using user ID":
        sup_id.user_id()


