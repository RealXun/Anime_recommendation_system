import streamlit as st
import os
import sys
from utils import recommend
from PIL import Image
import pickle
import requests
from pathlib import Path
from streamlit_option_menu import option_menu
from PIL import Image
import requests
from io import BytesIO
from body import testing
from body import un_based_rate
from body import un_based_feat
from body import sup_id
from body import about
from body import about_me


def super_ratings_based(id,n,genre,type, method):
    if method == "and":
        similar_animes = recommend.create_dict_su(recommend.sort_it(id), genre, type,method,n)
    else:
        similar_animes = recommend.create_dict_su(recommend.sort_it(id),genre,type,method,n)
    
    return similar_animes
    

def results(users_id,number_of_recommendations,selected_genre,selected_type, method):
    result = super_ratings_based(users_id,number_of_recommendations,selected_genre,selected_type, method)
    if result is not None: 
        # If the recommendation results are not empty, create a new dictionary to store them
        new_dict={}
        for di in result:
            # For each recommendation, add a new entry to the dictionary using the anime name as the key
            new_dict[di['name']]={}
            # Copy all other properties of the recommendation into the dictionary entry for that anime
            for k in di.keys():
                if k =='name': continue
                new_dict[di['name']][k]=di[k]
        # Determine how many rows and columns are needed to display the recommendations
        num_cols = 5
        num_rows = len(result) // num_cols + 1
        # Loop through each row of recommendations
        for row_idx in range(num_rows):
            # Create a new set of columns to display each recommendation
            cols = st.columns(num_cols)
            # Loop through each column and get the key (anime name) for that column's recommendation
            for col_idx, key in enumerate(list(new_dict.keys())[row_idx*num_cols:(row_idx+1)*num_cols]):
                # Get the recommendation for the current anime
                result = new_dict[key]
                # Get the cover image for the anime from the recommendation data
                response = requests.get(result['cover'])
                img = Image.open(BytesIO(response.content))
                # Display the anime information in a container within the current column
                with cols[col_idx].container():
                    cols[col_idx].image(img, use_column_width=True)
                    cols[col_idx].write(f"**{result['english_title']}**")
                    if 'japanese_title' in result:
                        cols[col_idx].write(f"**{result['japanese_title']}")
                    if 'type' in result:
                        cols[col_idx].write(f"**Type:** {result['type']}")
                    if 'episodes' in result:
                        cols[col_idx].write(f"**Episodes:** {result['episodes']}")
                    if 'duration' in result:
                        cols[col_idx].write(f"**Duration:** {result['duration']}")
                    if 'rating' in result:
                        cols[col_idx].write(f"**Rating:** {result['rating']}")
                    if 'score' in result:
                        cols[col_idx].write(f"**Score:** {result['score']}/10")
                    # Display the estimated score for the recommendation
                    cols[col_idx].write(f"**{float(result['Estimate_Score'])}**")
    else:
        # If there are no recommendations to display, inform the user
        st.write("Sorry, there is no matches for this, try again with different filters.")
        
    # If the user has not entered enough information to get recommendations, prompt them to do so
    if not (users_id and number_of_recommendations):
        st.write("Please enter anime name and number of recommendations to get the recommendation.")


