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
from scripts import un_based_rate
from scripts import un_based_feat
from scripts import sup_id
from scripts import about

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




def it_is_about():
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Anime Recommendation systems based on Data from MyAnimeList**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("The goal of this project to create 3 types of Anime recommendation system.\n\
             But, what is a Anime recommendation system? is a type of recommendation system \
             that is specifically designed to suggest Anime titles to users based on their \
             preferences. This system uses various algorithms and data analysis techniques \
             to analyze user behavior, interests, and interactions with different Anime titles,\
              and then recommends titles that the user may be interested in watching.\n\
             The system typically works by analyzing a user's viewing history and rating\
              history to determine their preferences. It may also consider other factors\
              such as the user's demographic information, the popularity of the Anime title,\
              and the similarity between different Anime titles.\n\
             Once the system has analyzed this data, it generates a list of recommended Anime \
             titles for the user to watch. These recommendations may be based on user ratings \
             and viewing habits, as well as other factors such as the similarity between different\
              Anime titles or the popularity of a particular title.")

    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Exploratoy Data Analysis</p>', unsafe_allow_html=True)
    st.write("Exploratory Data Analysis (EDA) is an important step in the machine learning pipeline because\
              it helps you understand the data you are working with. EDA is the process of exploring and\
              summarizing the main characteristics of a dataset in order to gain insights and identify patterns\
              that may be useful in creating a machine learning model.\
             \nSome reasons why doing EDA before creating a machine learning model is important include:\n\
             \n     - Identify data quality issues\
             \n     - Understand the relationships between variables\
             \n     - Identify patterns and trends\
             \n     - Visualize the data\
             \n\nIn summary, EDA is an important step in the machine learning pipeline because it helps you\
             understand the data you are working with, identify data quality issues, and identify patterns\
             and trends that may be useful in creating a machine learning model.")
    with st.expander("See the process."):    
        path_to_html = files_folder + "/" + "EDA_Anime.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=83650)


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
        path_to_html = files_folder + "/" + "unsupervised_user_explicit_rating_based_recommendation_system.html" 
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
        path_to_html = files_folder + "/" + "unsupervised_content_based_filtered_filtered.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=10600)



    st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Supervised Collaborative Filtering based on ratings Using SVD method</p>', unsafe_allow_html=True)
    st.write("Supervised Collaborative Filtering based on ratings is a recommendation system method that predicts user preferences\
              based on the ratings of similar users, along with additional data sources. It is called supervised because it relies\
              on a training dataset to learn the patterns of user behavior.\
             \n\nThe user-item matrix in Collaborative Filtering represents the users' ratings for various items. In Supervised\
              Collaborative Filtering based on ratings, the model is trained on the historical data of user-item interactions,\
              along with additional data sources such as demographic information, search queries, or purchase history. The model\
              learns the patterns of user behavior and uses this information to predict user preferences for items that they have\
             not yet interacted with.\
             \n\nSingular Value Decomposition (SVD) is a matrix factorization technique used in Collaborative Filtering to reduce\
              the dimensionality of the user-item matrix. SVD can decompose a large matrix into smaller matrices that capture the\
              underlying relationships between users and items. In Supervised Collaborative Filtering based on ratings using SVD,\
              the user-item matrix is decomposed into three smaller matrices: U, S, and V. U represents the users' preferences,\
              S represents the singular values, and V represents the items' features.\
             \n\nThe model is trained on a training dataset and evaluated on a testing dataset. The performance is evaluated using\
              metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Precision, Recall, and F1-score.\
              The SVD-based Supervised Collaborative Filtering method can be implemented using libraries such as Surprise in Python. As we do below")
    with st.expander("See the process."):
        path_to_html = files_folder + "/" + "supervised_user_based_collaborative_filtering.html" 
        # Read file and keep in variable
        with open(path_to_html,'r', encoding='utf-8') as f: 
            html_data = f.read()
            st.components.v1.html(html_data,height=23000)