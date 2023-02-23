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
    st.write("Collaborative Filtering is a technique used in recommendation systems, which aims to predict user preferences\
        based on their historical behavior or preferences. In anime recommendation systems, Collaborative Filtering can be \
        used to recommend new anime to users based on their ratings and preferences.\
        \n\nk-Nearest Neighbors (kNN) is a popular algorithm used in Collaborative Filtering. \
        The kNN algorithm works by finding the k most similar users to a target user based \
        on their ratings. Once the k most similar users have been identified, \the algorithm recommends anime that have high ratings among those users.")
    with st.expander("See the process."):
        path_to_html = body_folder + "/" + "unsupervised_user_explicit_rating_based_recommendation_system.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=8300)  

       

    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Unsupervised content based recommendation system</p>', unsafe_allow_html=True)
    st.write("An unsupervised content-based recommendation system is a type of recommendation system that uses the features of items to recommend similar items to users.\
         This approach is unsupervised because it doesn't require explicit feedback from users to make recommendations.\
        \n\nThe basic idea behind a content-based recommendation system is to analyze the attributes or characteristics of items (such as movies, music, or books)\
        and then recommend items that are similar to those that a user has already shown interest in. For example, if a user likes action movies,\
        the recommendation system might recommend other action movies with similar characteristics, such as fast-paced plots and explosive special effects.\
        \n\nTo create a content-based recommendation system, the first step is to gather data about the items being recommended. This data might include \
        information such as the genre, actors, director, release date, and plot summary for movies or the artist, album, genre, and song lyrics for music.\
        \n\nOnce the data has been collected, the next step is to analyze it to identify patterns and similarities between items. This can be done using\
        machine learning techniques such as clustering, dimensionality reduction, or classification. The resulting model can then be used to recommend\
        items to users based on their preferences.\
        \n\nOne advantage of a content-based recommendation system is that it can work well even with sparse or incomplete user data, since it doesn't \
        rely on user feedback to make recommendations. However, it may be less effective in situations where users have diverse interests or where there\
        are not enough attributes to accurately capture the essence of the items being recommended.")
    with st.expander("See the process."):
        path_to_html = body_folder + "/" + "unsupervised_content_based_filtered_filtered.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=10600)



    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Supervised Collaborative Filtering based on ratings Using SVD method</p>', unsafe_allow_html=True)
    st.write("Supervised collaborative filtering based on ratings using the SVD (Singular Value Decomposition) method involves using a labeled dataset of\
        user ratings to train a model that can predict the ratings that a user would give to items that they haven't yet rated.")
    with st.expander("See the process."):
        path_to_html = body_folder + "/" + "supervised_user_based_collaborative_filtering.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=11800)