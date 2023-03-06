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
import math

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
data_folder = (PROJECT_ROOT + "/" + "data")
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")

saved_models_folder = (data_folder + "/" + "saved_models")
raw_data = (data_folder + "/" + "_raw")
processed_data = (data_folder + "/" + "processed")
content_based_supervised_data = (data_folder + "/" + "processed" + "/" + "content_based_supervised")
images = (PROJECT_ROOT + "/" + "images")
cover_images = (images + "/" + "Cover_images")



def info():
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Find information from an anime**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)

    selected_name = st.selectbox("Choose the anime name",sorted(recommend.names_unique()))
    result = recommend.all_anime_dict()
    # Create an empty dictionary to store the recommendations
    new_dict = {}

    # Loop through the recommendations and add each one to the dictionary
    for anime in result:
        if anime['english_title'] == selected_name:
            # For each recommendation, add a new entry to the dictionary using the anime name as the key
            new_dict[anime['english_title']]={}
            # Copy all other properties of the recommendation into the dictionary entry for that anime
            for k in anime.keys():
                if k =='name': continue
                new_dict[anime['english_title']][k] = anime[k]

    # Set up the layout with two columns
    col1, col2 = st.columns([1, 2])

    # Display the cover image in the left column
    with col1:
        
        if new_dict[selected_name]['cover']: 
            response = requests.get(new_dict[selected_name]['cover'])     
            img = Image.open(BytesIO(response.content))
            st.image(img, use_column_width=True)

    # Display the anime information in the right column
    with col2:
        if new_dict[selected_name]['english_title']:
            st.write(f"**English name:** {new_dict[selected_name]['english_title']}")

        if new_dict[selected_name]['romanji']:
            st.write(f"**Romanji name:** {new_dict[selected_name]['romanji']}")

        if new_dict[selected_name]['japanses_title']:
            st.write(f"**Jananese name:** {new_dict[selected_name]['japanses_title']}")

        if new_dict[selected_name]['genre']:
            st.write(f"**Genre:** {new_dict[selected_name]['genre']}")

        if new_dict[selected_name]['type']:
            st.write(f"**Type:** {new_dict[selected_name]['type']}")

        if new_dict[selected_name]['source']:
            st.write(f"**Source:** {new_dict[selected_name]['source']}")

        if new_dict[selected_name]['released']:
            st.write(f"**Released Year:** {new_dict[selected_name]['released']}") 

        if (new_dict[selected_name]['duration']):
            st.write(f"**Duration:** {new_dict[selected_name]['duration']}")
        else:
            st.write("**Duration:** N/A")

        if (new_dict[selected_name]['episodes']):
            st.write(f"**Nº of episodes:** {new_dict[selected_name]['episodes']}")
        else:
            st.write("**Nº of episodes:** N/A")

        if (new_dict[selected_name]['rating']):
            st.write(f"**Rating:** {new_dict[selected_name]['rating']}")

        if not math.isnan(new_dict[selected_name]['score']):
            st.write(f"**Score:** {new_dict[selected_name]['score']}/10")
        else:
            st.write("**Score:** N/A")

        if not math.isnan(new_dict[selected_name]['rank']):
            st.write(f"**Ranking:** {new_dict[selected_name]['rank']}")
        else:
            st.write("**Ranking:** N/A")

        if new_dict[selected_name]['synopsis']:
            st.write(f"**Synopsis:** {new_dict[selected_name]['synopsis']}")

        if new_dict[selected_name]['anime_id']:               
            st.write(f"**Id in MyAnimelist Website:** {new_dict[selected_name]['anime_id']}")