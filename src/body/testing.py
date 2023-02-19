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

def test_it():
    #Add the cover image for the cover page. Used a little trick to center the image
                # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.title('Anime Recommendation System')
    st.title('Unsupervised user based collaborative filtering')
    # Get the user's favorite movie
    to_search = st.text_input("Enter a string:")
    if to_search:
        if to_search.isnumeric():
            st.write("Input contains only numbers. Please enter a string with at least one non-numeric character.")
        else:
            st.write(f"Input is valid: {to_search}")
    # Get the user's favorite movie
    number_of_recommendations = st.slider('How many recommendations would you like to get?', min_value=1, max_value=100, value=5, step=1)
    st.write('The current number is ', number_of_recommendations)
    def testing(name,genre,type,n):
        similar_animes = recommend.create_dict(recommend.unsupervised_user_based_recommender(name),genre,type,n)
        return similar_animes
    ## Drop down menu to select the genre
    option_gere = st.selectbox('What kind of genre would you like to search (you can choose all)',('All','Email', 'Home phone', 'Mobile phone'))
    st.write('You selected:', option_gere)
    ## Drop down menu to select the type
    option_type = st.selectbox('What type of anime would you like to search (you can choose all)',('All','TV', 'OVA', 'ONA'))
    st.write('You selected:', option_type)
    if (st.button('Get the Recommendation')):
        # dataframe = load('../models/df.pkl')
        result = testing(to_search,option_gere,option_type,number_of_recommendations)
        new_dict={}
        for di in result:
            new_dict[di['name']]={}
            for k in di.keys():
                if k =='name': continue
                new_dict[di['name']][k]=di[k]
                
        num_cols = 3
        num_rows = len(result) // num_cols + 1
        for row_idx in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, key in enumerate(list(new_dict.keys())[row_idx*num_cols:(row_idx+1)*num_cols]):
                result = new_dict[key]
                #cols[col_idx].image(result['cover_image'], width=200)
                cols[col_idx].write(f"{result['english_title']}")
                cols[col_idx].write(f"{result['japanses_title']}")
                #url = cols[col_idx].write(f"{result['img']}")
                #cols[col_idx].write(f"{result['cover']}")
                # Fetch image from URL
                response = requests.get(result['cover'])
                img = Image.open(BytesIO(response.content))
                
                # Display image, title, and rating
                cols[col_idx].image(img, use_column_width=True)
                cols[col_idx].write(f"{result['type']}, Episodes: {int(result['episodes'])}")
                cols[col_idx].write(f"{result['duration']}")
                cols[col_idx].write(f"{result['rating']}")
                cols[col_idx].write(f"Score: {result['score']}/10")