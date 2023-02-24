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
import joblib


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




def explore_data():
    import streamlit as st
    import pandas as pd

    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)

    st.markdown('<p class="font">Hi there!!👋 Take a look at my repositories and let\'s get in touch!</p>', unsafe_allow_html=True)

    # Opening cleaned df using Pickle
    anime_df = joblib.load(raw_data + "/" + "anime_eda.pkl")
    st.dataframe(anime_df)

    st.title('My Streamlit App')

    # Create a selectbox for choosing the column to display

    columns = ["Type","Source","Rating","Genre","Theme","Released","Studios","Producers"]    
    column = st.selectbox('Select a column', columns)
    if column == "Theme":
        split_values = anime_df["Theme"].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
        # Drop rows containing the value "Unknown"
        split_values = split_values[split_values != "Unknown"]        
        # Count the frequency of each unique value
        value_counts = split_values.value_counts().head(30)
    elif column == "Genre":
        split_values = anime_df["Genre"].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
        # Drop rows containing the value "Unknown"
        split_values = split_values[split_values != "Unknown"]        
        # Count the frequency of each unique value
        value_counts = split_values.value_counts().head(30)
    elif column == "Theme":
        split_values = anime_df["Theme"].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
        # Drop rows containing the value "Unknown"
        split_values = split_values[split_values != "Unknown"]        
        # Count the frequency of each unique value
        value_counts = split_values.value_counts().head(30)
    elif column == "Studios":
        split_values = anime_df["Studios"].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
        # Drop rows containing the value "Unknown"
        split_values = split_values[split_values != "Unknown"]        
        # Count the frequency of each unique value
        value_counts = split_values.value_counts().head(30)
    elif column == "Producers":
        split_values = anime_df["Producers"].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
        # Drop rows containing the value "Unknown"
        split_values = split_values[split_values != "Unknown"]        
        # Count the frequency of each unique value
        value_counts = split_values.value_counts().head(30).sort_index(ascending=False)
        # Sort the resulting series object in descending order
        value_counts = value_counts.sort_values(ascending=False)
    else:
        # Count the frequency of each value in the selected column
        value_counts = anime_df[column].value_counts().sort_values(ascending=False)
    
    # Create a bar chart
    st.bar_chart(value_counts)
